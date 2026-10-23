"""Explicit, reproducible binaural scene rendering for WavJEPA-Nat.

The convolution, noise aggregation and fade recipe adapts the BSD-3-Clause
WavJEPA implementation at commit ``6f4e1b1f18316c50768c8523f325ccffeaa6985b``:
``data_modules/scene_module/generate_scenes.py`` and
``hear_api/heaRIR/scene_module/generate_scenes.py``. Copyright (c) 2024,
PierreGtch; the retained license is ``src/vendor/wavjepa/LICENSE``. We replace
torchaudio's FFT convolution with torch.fft, use an explicit local generator,
validate assets, preserve short-noise reverberation tails, and use one SNR gain
for both ears instead of independently changing their spatial level cues.

Section 3.3 / Appendix D of arXiv:2509.23238v1 specify matched source/noise
BRIRs, 50% localized (one noise BRIR), 50% diffuse (three to five noise BRIRs),
5--40 dB SNR, and 200 ms noise fades. The manifest supplies these existing
SoundSpaces 2.0 / Matterport3D scenes and WHAMR! noise; this module does not
generate BRIRs, download assets, resample them, or substitute dry audio.

Manifest JSON (paths may be relative to its directory)::

    {"schema_version": 1, "sample_rate": 16000, "split": "train",
     "scenes": [{"id": "room-scene", "noise_type": "localized",
                 "source_rir": "source.npy", "noise_rirs": ["noise.npy"]},
                {"id": "other-scene", "noise_type": "diffuse",
                 "source_rir": "source2.npy",
                 "noise_rirs": ["n1.npy", "n2.npy", "n3.npy"]}],
     "noise_files": ["whamr.wav"],
     "asset_sha256": {"source.npy": "<sha256>", "noise.npy": "<sha256>",
                      "source2.npy": "<sha256>", "n1.npy": "<sha256>",
                      "n2.npy": "<sha256>", "n3.npy": "<sha256>",
                      "whamr.wav": "<sha256>"}}

NPY BRIRs must contain a numeric [2, samples] array already resampled to the
manifest's declared rate. WAV BRIRs must be stereo PCM16; noise WAVs must be
mono PCM16, with headers matching 16 kHz. An optional ``provenance`` object
records dataset revisions and preprocessing. Production requires a complete
``asset_sha256`` mapping; hashes are verified lazily on first use and cached
only while the file stat is unchanged. Train/evaluation splits need
separate manifests; splitting by house is the caller's responsibility.

Render a complete recording before selecting chronological windows. Cropping
the mono signal first would remove the reverberation history at crop borders.

For a large private S3 corpus, replace ``asset_sha256`` with ``bucket``,
``endpoint``, and ``assets``. Each logical path maps to
``{"key": "nat/scene.npz", "bytes": 123, "sha256": "...", "index": 0}``.
Logical NPY/NPZ paths may share a packed [1..6, 2, samples] scene object and
select its source/noise rows with ``index``. NPZ uses array ``arr_0`` by default;
``array_key`` explicitly selects a different member. PCM noise has no index.
The caller supplies a separate cache directory/budget for every consumer.
Downloads reuse the AudioSet loader's bounded, checksum-verified S3 cache;
remote paths are never silently replaced by local files.
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class _Scene:
    identity: str
    noise_type: str
    source_rir: Path
    noise_rirs: tuple[Path, ...]


@dataclass(frozen=True)
class RenderedScene:
    """Rendered mixture and its components, after a shared gain when normalized."""

    waveform: Tensor
    source: Tensor
    noise: Tensor
    metadata: dict[str, Any]


def generator_for_clip(seed: int, clip_id: str, epoch: int) -> torch.Generator:
    """Local randomness independent of worker scheduling, ranks and global RNG."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError("clip_id must be a nonempty string")
    payload = json.dumps([seed, clip_id, epoch], separators=(",", ":")).encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return torch.Generator(device="cpu").manual_seed(value)


def convolve_with_rir(waveform: Tensor, rir: Tensor) -> Tensor:
    """Causal full FFT convolution, retaining the original recording duration.

    This is the upstream per-ear full-convolution-and-truncate operation. Both
    ears share one FFT size; no channel-specific normalization is performed.
    """
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("source waveform must be nonempty mono [samples]")
    if rir.ndim != 2 or rir.shape[0] != 2 or rir.shape[1] == 0:
        raise ValueError("BRIR must have shape [2, nonzero samples]")
    if not torch.isfinite(waveform).all() or not torch.isfinite(rir).all():
        raise ValueError("waveform and BRIR must be finite")
    full_length = waveform.numel() + rir.shape[-1] - 1
    fft_size = 1 << (full_length - 1).bit_length()
    spectrum = torch.fft.rfft(waveform, n=fft_size)
    rir_spectrum = torch.fft.rfft(rir, n=fft_size)
    return torch.fft.irfft(spectrum * rir_spectrum, n=fft_size)[..., : waveform.numel()]


def fade_noise(noise: Tensor, sample_rate: int = 16000, duration: float = 0.2) -> Tensor:
    """Apply Appendix D's two linear fades, safely handling short recordings."""
    if noise.ndim != 1 or noise.numel() == 0:
        raise ValueError("noise must be nonempty mono [samples]")
    result = noise.clone()
    length = min(round(duration * sample_rate), result.numel() // 2)
    if length:
        result[:length] *= torch.linspace(0, 1, length, dtype=result.dtype)
        result[-length:] *= torch.linspace(1, 0, length, dtype=result.dtype)
    return result


def _read_pcm(path: Path, sample_rate: int, channels: int, max_samples: int) -> Tensor:
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != channels
                or handle.getsampwidth() != 2
                or handle.getcomptype() != "NONE"
            ):
                raise ValueError(f"Expected {channels}-channel uncompressed PCM16 WAV: {path}")
            if handle.getframerate() != sample_rate:
                raise ValueError(f"Asset sample rate must be {sample_rate} Hz: {path}")
            frames = min(handle.getnframes(), max_samples)
            payload = handle.readframes(frames)
            if len(payload) != frames * channels * 2 or frames == 0:
                raise ValueError(f"Empty or truncated PCM asset: {path}")
    except (OSError, wave.Error) as error:
        raise ValueError(f"Cannot read PCM asset {path}: {error}") from error
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples.reshape(-1, channels).T.copy())


class NaturalisticAudio:
    """Render paired, precomputed binaural scenes using explicit asset manifests.

    Defaults are fully naturalistic, with no clean examples. ``clean_probability``
    is an explicit ablation, never an error recovery path. One scene and one
    WHAMR! clip are drawn per original recording, before any training crops.
    ``snr_db`` applies to energy summed over both ears and the complete clip.
    A silent/DC-only source remains silent with zero noise gain, the continuous
    zero-source limit of the upstream mixer. Its achieved SNR is undefined;
    metadata records this explicitly without dropping a distributed sample.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        snr_db: tuple[float, float] = (5.0, 40.0),
        diffuse_probability: float = 0.5,
        clean_probability: float = 0.0,
        normalize: bool = True,
        max_audio_seconds: float = 30.0,
        max_rir_seconds: float = 3.0,
        require_asset_checksums: bool = False,
        cache_dir: str | Path | None = None,
        cache_max_gib: float = 1.0,
        cache_min_free_gib: float = 0.0,
        download_timeout_seconds: float = 120.0,
    ) -> None:
        if isinstance(sample_rate, bool) or sample_rate != 16000:
            raise ValueError("WavJEPA-Nat assets and source audio must use sample_rate=16000")
        if (
            len(snr_db) != 2
            or not all(math.isfinite(float(value)) for value in snr_db)
            or not 5 <= float(snr_db[0]) <= float(snr_db[1]) <= 40
        ):
            raise ValueError("snr_db must be an ordered pair within [5, 40] dB")
        for name, probability in (
            ("diffuse_probability", diffuse_probability),
            ("clean_probability", clean_probability),
        ):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if not math.isfinite(max_audio_seconds) or not 0 < max_audio_seconds <= 30:
            raise ValueError("max_audio_seconds must be positive and at most 30")
        if not math.isfinite(max_rir_seconds) or not 0 < max_rir_seconds <= 3:
            raise ValueError("max_rir_seconds must be positive and at most 3")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        try:
            content = self.manifest_path.read_bytes()
            manifest = json.loads(content)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read naturalistic asset manifest: {self.manifest_path}"
            ) from error
        self.manifest_sha256 = hashlib.sha256(content).hexdigest()
        if not isinstance(manifest, dict):
            raise ValueError("Naturalistic manifest must be a JSON object")
        if manifest.get("schema_version") != 1 or isinstance(manifest.get("schema_version"), bool):
            raise ValueError("Naturalistic manifest requires schema_version=1")
        if manifest.get("sample_rate") != sample_rate:
            raise ValueError("Naturalistic manifest sample_rate must equal 16000")
        if manifest.get("split") not in ("train", "val", "test"):
            raise ValueError("Naturalistic manifest must declare split=train, val, or test")
        self.split = manifest["split"]
        self.provenance = manifest.get("provenance", {})
        if not isinstance(self.provenance, dict):
            raise ValueError("Naturalistic manifest provenance must be an object")
        self.sample_rate = sample_rate
        self.snr_db = tuple(float(value) for value in snr_db)
        self.diffuse_probability = float(diffuse_probability)
        self.clean_probability = float(clean_probability)
        self.normalize = bool(normalize)
        self.max_audio_samples = round(max_audio_seconds * sample_rate)
        self.max_rir_samples = round(max_rir_seconds * sample_rate)
        self.remote_assets = manifest.get("assets") is not None
        self.assets: dict[Path, dict[str, Any]] = {}
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        self._cache = None
        for name, value, allow_zero in (
            ("cache_max_gib", cache_max_gib, False),
            ("cache_min_free_gib", cache_min_free_gib, True),
            ("download_timeout_seconds", download_timeout_seconds, False),
        ):
            if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
                raise ValueError(
                    f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
                )
        self.cache_budget = int(cache_max_gib * 1024**3)
        self.cache_minimum_free = int(cache_min_free_gib * 1024**3)
        self.download_timeout_seconds = float(download_timeout_seconds)
        if self.remote_assets:
            self._parse_remote_assets(manifest)
        scenes = manifest.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("Naturalistic manifest requires nonempty scenes")
        by_kind: dict[str, list[_Scene]] = {"localized": [], "diffuse": []}
        identities: set[str] = set()
        for record in scenes:
            if not isinstance(record, dict):
                raise ValueError("Each scene must be an object")
            identity, kind = record.get("id"), record.get("noise_type")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Each scene must have a unique nonempty id")
            if kind not in by_kind:
                raise ValueError("Scene noise_type must be localized or diffuse")
            paths = record.get("noise_rirs")
            if not isinstance(paths, list) or (
                len(paths) != 1 if kind == "localized" else len(paths) not in (3, 4, 5)
            ):
                raise ValueError(
                    "Localized scenes require 1 noise BRIR; diffuse scenes require 3-5"
                )
            source_path = self._asset_path(record.get("source_rir"), rir=True)
            noise_paths = tuple(self._asset_path(path, rir=True) for path in paths)
            if len(set(noise_paths)) != len(noise_paths):
                raise ValueError("Diffuse scene noise BRIR paths must be distinct")
            if self.remote_assets:
                locations = {
                    (
                        self.assets[path]["key"],
                        self.assets[path].get("index"),
                        self.assets[path]["array_key"],
                    )
                    for path in noise_paths
                }
                if len(locations) != len(noise_paths):
                    raise ValueError("Diffuse noise BRIRs must select distinct remote locations")
            by_kind[kind].append(_Scene(identity, kind, source_path, noise_paths))
            identities.add(identity)
        for kind, required in (
            ("localized", diffuse_probability < 1),
            ("diffuse", diffuse_probability > 0),
        ):
            if required and not by_kind[kind]:
                raise ValueError(f"Manifest has no {kind} scenes for the requested probability")
        self.scenes = {kind: tuple(values) for kind, values in by_kind.items()}
        files = manifest.get("noise_files")
        if not isinstance(files, list) or not files:
            raise ValueError("Naturalistic manifest requires nonempty WHAMR! noise_files")
        self.noise_files = tuple(self._asset_path(path, rir=False) for path in files)
        # Bind asset contents, not just filesystem paths, into reproducible manifests.
        # Validate coverage now; hash lazily on actual use rather than reading the
        # entire BRIR corpus in every DataLoader worker during construction.
        checksums = manifest.get("asset_sha256")
        if require_asset_checksums and not isinstance(checksums, dict) and not self.remote_assets:
            raise ValueError("Naturalistic training requires a complete asset_sha256 mapping")
        self.asset_sha256: dict[Path, str] = {}
        if checksums is not None:
            if not isinstance(checksums, dict):
                raise ValueError("asset_sha256 must be an object mapping asset paths to SHA-256")
            for name, digest in checksums.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError("asset_sha256 requires paths and lowercase SHA-256 digests")
                asset = Path(name).expanduser()
                if not asset.is_absolute():
                    asset = self.manifest_path.parent / asset
                asset = asset.resolve()
                if asset in self.asset_sha256:
                    raise ValueError("asset_sha256 contains duplicate physical paths")
                self.asset_sha256[asset] = digest
            referenced = set(self.noise_files)
            for group in self.scenes.values():
                for scene in group:
                    referenced.add(scene.source_rir)
                    referenced.update(scene.noise_rirs)
            if set(self.asset_sha256) != referenced:
                raise ValueError(
                    "asset_sha256 must cover exactly all referenced BRIR and noise assets"
                )
        self._verified_assets: OrderedDict[Path, tuple[int, int, int, int, int]] = OrderedDict()
        if self.remote_assets and any(
            digest != self.assets[path]["sha256"] for path, digest in self.asset_sha256.items()
        ):
            raise ValueError("Local checksum mapping contradicts remote asset SHA-256")
        if self.remote_assets:
            referenced = set(self.noise_files)
            for group in self.scenes.values():
                for scene in group:
                    referenced.add(scene.source_rir)
                    referenced.update(scene.noise_rirs)
            if referenced != set(self.assets):
                raise ValueError(
                    "Remote assets must cover exactly all referenced BRIR and noise paths"
                )

    def _parse_remote_assets(self, manifest: dict[str, Any]) -> None:
        self.bucket, self.endpoint = manifest.get("bucket"), manifest.get("endpoint")
        if not isinstance(self.bucket, str) or not self.bucket or "/" in self.bucket:
            raise ValueError("Remote naturalistic manifest requires an explicit bucket")
        parsed = urlparse(self.endpoint) if isinstance(self.endpoint, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError(
                "Remote naturalistic endpoint must be HTTPS without credentials or query"
            )
        assets = manifest["assets"]
        if not isinstance(assets, dict) or not assets:
            raise ValueError("Remote assets must be a nonempty object")
        objects: dict[str, tuple[str, int]] = {}
        for name, record in assets.items():
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
                or not isinstance(record, dict)
            ):
                raise ValueError("Remote asset names must be safe relative logical paths")
            key, digest, size = record.get("key"), record.get("sha256"), record.get("bytes")
            if (
                not isinstance(key, str)
                or not key
                or key.startswith("/")
                or ".." in PurePosixPath(key).parts
                or "\\" in key
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(size) is not int
                or not 0 < size <= self.cache_budget
            ):
                raise ValueError(
                    "Remote assets require safe key, SHA-256, and bytes fitting the cache"
                )
            index = record.get("index")
            if index is not None and (type(index) is not int or not 0 <= index < 6):
                raise ValueError("Packed BRIR index must be an integer in [0, 5]")
            suffix = Path(name).suffix.lower()
            if index is not None and suffix not in (".npy", ".npz"):
                raise ValueError("Packed BRIR index is valid only for NPY/NPZ assets")
            array_key = record.get("array_key", "arr_0")
            if not isinstance(array_key, str) or not array_key or "/" in array_key:
                raise ValueError("NPZ array_key must name one array")
            if key in objects and objects[key] != (digest, size):
                raise ValueError(
                    "Repeated remote object keys must have identical SHA-256 and bytes"
                )
            objects[key] = (digest, size)
            logical = (self.manifest_path.parent / name).resolve()
            if logical in self.assets:
                raise ValueError("Duplicate remote logical asset path")
            self.assets[logical] = {**record, "source": name, "array_key": array_key}

    def _asset_path(self, value: Any, *, rir: bool) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Asset paths must be nonempty strings")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        path = path.resolve()
        if self.remote_assets and path not in self.assets:
            raise ValueError(f"Naturalistic logical asset is missing from remote assets: {value}")
        if not self.remote_assets and not path.is_file():
            raise ValueError(f"Naturalistic asset is missing: {path}")
        allowed = (".npy", ".npz", ".wav") if rir else (".wav",)
        if path.suffix.lower() not in allowed:
            raise ValueError(f"Expected {'NPY or WAV BRIR' if rir else 'WAV noise'}: {path}")
        return path

    def _load_rir(self, path: Path) -> Tensor:
        physical = self._materialize(path)
        if path.suffix.lower() == ".wav":
            with wave.open(str(physical), "rb") as handle:
                if handle.getnframes() > self.max_rir_samples:
                    raise ValueError(f"BRIR exceeds configured duration: {path}")
            rir = _read_pcm(physical, self.sample_rate, 2, self.max_rir_samples)
        else:
            try:
                # Read-only mmap inspects shape before materializing bounded float32 data.
                asset = self.assets.get(path, {})
                if path.suffix.lower() == ".npz":
                    array_key = asset.get("array_key", "arr_0")
                    with zipfile.ZipFile(physical) as archive:
                        member = archive.getinfo(f"{array_key}.npy")
                        if member.file_size > 6 * 2 * self.max_rir_samples * 8 + 65536:
                            raise ValueError("NPZ array exceeds the bounded BRIR shape")
                    with np.load(physical, allow_pickle=False) as archive:
                        array = archive[array_key]
                else:
                    array = np.load(physical, mmap_mode="r", allow_pickle=False)
                index = asset.get("index")
                if index is not None:
                    if array.ndim != 3 or not 0 < array.shape[0] <= 6 or index >= array.shape[0]:
                        raise ValueError("Packed BRIR asset must have shape [1..6, 2, samples]")
                    array = array[index]
                if (
                    array.ndim != 2
                    or array.shape[0] != 2
                    or not 0 < array.shape[1] <= self.max_rir_samples
                    or array.dtype.kind not in "fiu"
                ):
                    raise ValueError(
                        f"BRIR must be numeric [2, samples] within duration bound: {path}"
                    )
                rir = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
            except (OSError, TypeError, ValueError, KeyError, zipfile.BadZipFile) as error:
                raise ValueError(f"Cannot read bounded binaural BRIR: {path}: {error}") from error
        if not torch.isfinite(rir).all() or not torch.count_nonzero(rir):
            raise ValueError(f"BRIR must be finite and nonzero: {path}")
        return rir

    def _materialize(self, path: Path) -> Path:
        if not self.remote_assets:
            self._verify_asset(path)
            return path
        if self.cache_dir is None:
            raise ValueError(
                "Remote naturalistic rendering requires an explicit per-consumer cache_dir"
            )
        if self._cache is None:
            # Reuse the tested private-S3 deadline, retries, size/SHA verification,
            # and pre-download eviction. Imports are lazy to avoid a module cycle.
            from src.data.sharded_audio_datamodule import _s3_client, _ShardCache

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            contract = {
                "manifest_sha256": self.manifest_sha256,
                "bucket": self.bucket,
                "endpoint": self.endpoint,
                "budget": self.cache_budget,
            }
            marker = self.cache_dir / "contract.json"
            if marker.exists():
                if json.loads(marker.read_text()) != contract:
                    raise ValueError("Naturalistic cache belongs to a different asset contract")
            elif any(self.cache_dir.iterdir()):
                raise ValueError("Refusing an unowned nonempty naturalistic cache directory")
            else:
                marker.write_text(json.dumps(contract))
            self._cache = _ShardCache(
                self.cache_dir,
                _s3_client(self.endpoint),
                self.bucket,
                self.cache_budget,
                self.cache_minimum_free,
                self.download_timeout_seconds,
            )
        return self._cache.get(self.assets[path])

    def _verify_asset(self, path: Path) -> None:
        """Verify selected content once per stable file version, with a bounded LRU."""
        if not self.asset_sha256:
            return
        stat = path.stat()
        fingerprint = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if self._verified_assets.get(path) == fingerprint:
            self._verified_assets.move_to_end(path)
            return
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.asset_sha256[path]:
            raise ValueError(f"Naturalistic asset SHA-256 mismatch: {path}")
        self._verified_assets[path] = fingerprint
        self._verified_assets.move_to_end(path)
        if len(self._verified_assets) > 1024:
            self._verified_assets.popitem(last=False)

    @staticmethod
    def _draw_index(length: int, generator: torch.Generator) -> int:
        return int(torch.randint(length, (), generator=generator).item())

    def render(self, waveform: Tensor, *, generator: torch.Generator) -> RenderedScene:
        """Render a complete mono recording before the caller extracts any crops."""
        if waveform.device.type != "cpu" or generator.device.type != "cpu":
            raise ValueError("Naturalistic data rendering requires CPU tensors and a CPU generator")
        if waveform.ndim != 1 or not 2 <= waveform.numel() <= self.max_audio_samples:
            raise ValueError("Naturalistic source must be mono [samples] within the duration bound")
        waveform = waveform.detach().to(torch.float32)
        if not torch.isfinite(waveform).all():
            raise ValueError("Naturalistic source must contain finite samples")
        waveform = waveform - waveform.mean()
        is_clean = self.clean_probability > 0 and bool(
            torch.rand((), generator=generator) < self.clean_probability
        )
        if is_clean:
            source = waveform.repeat(2, 1)
            noise = torch.zeros_like(source)
            metadata: dict[str, Any] = {"clean": True, "manifest_sha256": self.manifest_sha256}
        else:
            diffuse = bool(torch.rand((), generator=generator) < self.diffuse_probability)
            kind = "diffuse" if diffuse else "localized"
            choices = self.scenes[kind]
            scene = choices[self._draw_index(len(choices), generator)]
            noise_path = self.noise_files[self._draw_index(len(self.noise_files), generator)]
            mono_noise = _read_pcm(
                self._materialize(noise_path), self.sample_rate, 1, 10 * self.sample_rate
            )[0]
            # Appendix D trims the WHAMR! recording to 10 seconds before fades.
            if mono_noise.numel() > waveform.numel():
                offset = self._draw_index(mono_noise.numel() - waveform.numel() + 1, generator)
                mono_noise = mono_noise[offset : offset + waveform.numel()]
            mono_noise = fade_noise(mono_noise - mono_noise.mean(), self.sample_rate)
            start = self._draw_index(waveform.numel() - mono_noise.numel() + 1, generator)
            positioned_noise = torch.zeros_like(waveform)
            positioned_noise[start : start + mono_noise.numel()] = mono_noise
            source = convolve_with_rir(waveform, self._load_rir(scene.source_rir))
            # Place before convolving so short-noise tails survive after its dry end.
            noise = torch.zeros_like(source)
            for path in scene.noise_rirs:
                noise += convolve_with_rir(positioned_noise, self._load_rir(path))
            signal_energy, noise_energy = (
                source.double().square().sum(),
                noise.double().square().sum(),
            )
            if noise_energy <= 0:
                raise ValueError("Scene spatialized noise must have positive energy")
            snr = self.snr_db[0] + float(torch.rand((), generator=generator)) * (
                self.snr_db[1] - self.snr_db[0]
            )
            # One scalar preserves each signal's original interaural level difference.
            noise_scale = float(torch.sqrt(signal_energy / noise_energy * 10 ** (-snr / 10)))
            noise = noise * noise_scale
            metadata = {
                "clean": False,
                "manifest_sha256": self.manifest_sha256,
                "scene_id": scene.identity,
                "noise_type": kind,
                "noise_path": str(noise_path),
                "noise_source_count": len(scene.noise_rirs),
                "noise_start_samples": start,
                "noise_samples": mono_noise.numel(),
                "requested_snr_db": snr,
                "snr_db": snr if signal_energy > 0 else None,
                "source_degenerate": bool(signal_energy == 0),
                "noise_scale": noise_scale,
            }
        mixture = source + noise
        if self.normalize:
            # Use a single RMS gain across both ears, never per-ear centering or
            # standardization. The model owns joint crop instance normalization.
            scale = mixture.square().mean().sqrt().clamp_min(1e-8)
            source, noise, mixture = source / scale, noise / scale, mixture / scale
        if not torch.isfinite(mixture).all():
            raise ValueError("Naturalistic rendering produced nonfinite samples")
        return RenderedScene(mixture.contiguous(), source, noise, metadata)

    def __call__(self, waveform: Tensor, *, generator: torch.Generator) -> Tensor:
        return self.render(waveform, generator=generator).waveform

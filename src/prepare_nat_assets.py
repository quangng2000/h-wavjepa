"""Bounded, offline conversion of publisher-verified WavJEPA-Nat assets.

This command never downloads data, uploads files, or starts compute. Each local
source must match an explicitly supplied SHA-256 before any output is written.
BRIR shards contain the official GRAM scene arrays at 32 kHz; one scene is read
at a time, resampled to 16 kHz, and published as a compressed float32 NPZ. Only
exactly zero padded noise rows are removed. The complete three-second impulse
response is preserved. Output manifests are components for later assembly of a
training manifest, not evidence that the whole 70,000-scene corpus is ready.

WHAM noise conversion selects only ``wham_noise/tr/*.wav`` from a local ZIP.
The first recorded noise channel supplies the mono source, an explicit
preprocessing choice rather than a claimed paper requirement. Its duration is
preserved; scene rendering draws/trims/fades the noise later. A common peak gain
avoids clipping before PCM16 quantization and is recorded per asset.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import struct
import tarfile
import tempfile
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable
from urllib.parse import urlparse

import numpy as np

BRIR_REPOSITORY = "labhamlet/GRAM-Naturalistic-Scenes-Binaural"
BRIR_REVISION = "546f796991721befaffce60152af59bfd9ad64ff"
SOURCE_RATE = 32000
TARGET_RATE = 16000
MAX_RIR_SAMPLES = 3 * SOURCE_RATE
MAX_BRIR_MEMBER_BYTES = 6 * 2 * MAX_RIR_SAMPLES * 8 + 65536
MAX_NOISE_MEMBER_BYTES = 60 * TARGET_RATE * 2 * 8 + 65536


class PreparationLimit(RuntimeError):
    """A requested work bound was reached, leaving a partial manifest."""


def sha256_file(path: Path, check=lambda: None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            check()
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256")
    return value


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def _bounded_payload(handle: BinaryIO, size: int, maximum: int) -> bytes:
    if not 0 < size <= maximum:
        raise ValueError(f"Archive member size {size} exceeds bound {maximum}")
    content = handle.read(size + 1)
    if len(content) != size:
        raise ValueError("Truncated archive member")
    return content


def _bound_zip_directory(path: Path) -> None:
    """Bound ZIP central metadata before ZipFile constructs its member list."""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65557))
        tail = handle.read(65557)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise ValueError("Missing WHAM ZIP end-of-directory record")
    _, disk, directory_disk, disk_count, count, size, _, comment_size = struct.unpack(
        "<4s4H2LH", tail[position : position + 22]
    )
    if position + 22 + comment_size != len(tail):
        raise ValueError("Malformed WHAM ZIP end-of-directory record")
    # The official archive has fewer than 40,000 entries; ZIP64 offsets are
    # allowed, but larger/ZIP64 entry counts and directories are unnecessary.
    if disk or directory_disk or count != disk_count or count > 40000 or size > 64 * 1024**2:
        raise PreparationLimit("WHAM ZIP central directory exceeds bounded metadata")


def _npy_array(content: bytes) -> np.ndarray:
    """Validate the NPY header and exact payload size before NumPy allocates."""
    stream = io.BytesIO(content)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
    else:
        raise ValueError("Only NPY versions 1 and 2 are supported")
    if (
        len(shape) != 3
        or shape[:2] != (6, 2)
        or not 0 < shape[2] <= MAX_RIR_SAMPLES
        or dtype.kind != "f"
        or dtype.itemsize != 8
        or fortran
    ):
        raise ValueError("Official scene must be C-order float64 [6, 2, 1..96000]")
    expected = math.prod(shape) * dtype.itemsize
    if len(content) - stream.tell() != expected:
        raise ValueError("NPY payload length does not match its bounded shape")
    # No pickle or np.load allocation: header validation bounds this view.
    return np.frombuffer(content, dtype=dtype, offset=stream.tell()).reshape(shape)


def convert_scene_array(array: np.ndarray) -> tuple[np.ndarray, str, list[int]]:
    """Keep matched active BRIRs and apply an antialiasing polyphase filter."""
    if (
        array.ndim != 3
        or array.shape[:2] != (6, 2)
        or not 0 < array.shape[-1] <= MAX_RIR_SAMPLES
        or array.dtype.kind != "f"
        or array.dtype.itemsize != 8
        or not np.isfinite(array).all()
    ):
        raise ValueError("Scene must be finite float64 [6, 2, 1..96000]")
    if not np.any(array[0]):
        raise ValueError("Source BRIR must be nonzero")
    active = [index for index in range(1, 6) if np.any(array[index])]
    if len(active) not in (1, 3, 4, 5):
        raise ValueError("Scene needs one localized or three to five diffuse noise BRIRs")
    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise RuntimeError("Nat preparation requires scipy") from error
    indices = [0, *active]
    # Keep float64 through filtering, then store the bounded float32 result.
    converted = resample_poly(array[indices], up=1, down=2, axis=-1).astype(np.float32)
    if not np.isfinite(converted).all() or any(not np.any(row) for row in converted):
        raise ValueError("Resampling produced a nonfinite or empty BRIR")
    return converted, "localized" if len(active) == 1 else "diffuse", indices


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temp = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)


class _Output:
    def __init__(
        self,
        source: Path,
        expected_sha256: str,
        output_dir: Path,
        *,
        kind: str,
        max_output_bytes: int,
        max_seconds: float,
        external_check: Callable[[], None] | None = None,
    ) -> None:
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        self.deadline = time.monotonic() + max_seconds
        self.external_check = external_check
        self.maximum = max_output_bytes
        source = Path(source).expanduser().resolve()
        expected_sha256 = _digest(expected_sha256)
        before = source.stat()
        if not source.is_file() or sha256_file(source, self.check) != expected_sha256:
            raise ValueError("Source SHA-256 verification failed")
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Source changed during SHA-256 verification")
        self.source = source
        self.source_stat = after
        self.root = Path(output_dir).expanduser().resolve()
        # An explicit fresh destination prevents overwriting a previous attempt.
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "objects").mkdir()
        self.manifest = {
            "schema_version": 1,
            "kind": kind,
            "status": "partial",
            "sample_rate": TARGET_RATE,
            "split": "train",
            "source": {
                "name": source.name,
                "bytes": after.st_size,
                "sha256": expected_sha256,
                "verified": True,
            },
            "converted_count": 0,
            "output_bytes": 0,
            "scenes": [],
            "noise_files": [],
            "assets": {},
        }
        self.save()

    def check(self) -> None:
        if self.external_check is not None:
            self.external_check()
        if time.monotonic() >= self.deadline:
            raise PreparationLimit("Preparation runtime bound reached")

    def save(self) -> None:
        _atomic_json(self.root / "manifest.json", self.manifest)

    def publish(self, temp: Path, suffix: str) -> dict:
        self.check()
        size = temp.stat().st_size
        if self.manifest["output_bytes"] + size > self.maximum:
            raise PreparationLimit("Preparation output byte bound reached")
        digest = sha256_file(temp, self.check)
        relative = f"objects/{digest}{suffix}"
        destination = self.root / relative
        try:
            # Link atomically, without replacing an existing object.
            os.link(temp, destination)
        except FileExistsError:
            if destination.stat().st_size != size or sha256_file(destination) != digest:
                raise ValueError("Existing output object failed content verification")
        else:
            self.manifest["output_bytes"] += size
        return {"key": relative, "bytes": size, "sha256": digest}

    def finish(self) -> None:
        after = self.source.stat()
        if (after.st_ino, after.st_size, after.st_mtime_ns) != (
            self.source_stat.st_ino,
            self.source_stat.st_size,
            self.source_stat.st_mtime_ns,
        ):
            raise ValueError("Source changed during conversion")
        self.manifest["status"] = "complete"
        self.save()

    def failure(self, error: Exception) -> None:
        self.manifest["status"] = "partial" if isinstance(error, PreparationLimit) else "failed"
        self.manifest["reason"] = str(error)
        self.save()


def prepare_brir_shard(
    source: str | Path,
    expected_sha256: str,
    output_dir: str | Path,
    *,
    source_member: str,
    max_scenes: int = 1000,
    max_archive_members: int = 1100,
    max_output_bytes: int = 3 * 1024**3,
    max_seconds: float = 3600,
    external_check: Callable[[], None] | None = None,
) -> dict:
    """Convert one verified training shard and return its component manifest."""
    if re.fullmatch(r"train/webdataset_output/shard-\d{6}\.tar", source_member) is None:
        raise ValueError("source_member must name an official train/webdataset_output shard")
    if int(PurePosixPath(source_member).stem.split("-")[-1]) >= 70:
        raise ValueError("Official training BRIR shard index must be in [0, 69]")
    if type(max_scenes) is not int or not 0 < max_scenes <= 1000:
        raise ValueError("max_scenes must be in [1, 1000]")
    if type(max_archive_members) is not int or not 0 < max_archive_members <= 10000:
        raise ValueError("max_archive_members must be in [1, 10000]")
    output = _Output(
        Path(source),
        expected_sha256,
        Path(output_dir),
        kind="nat_brir_shard",
        max_output_bytes=max_output_bytes,
        max_seconds=max_seconds,
        external_check=external_check,
    )
    output.manifest["provenance"] = {
        "repository": BRIR_REPOSITORY,
        "revision": BRIR_REVISION,
        "path": source_member,
        "declared_license": "MIT",
        "source_sample_rate": SOURCE_RATE,
        "resampling": "scipy.signal.resample_poly(up=1, down=2, axis=-1), float64 then float32",
        "rir_duration": "preserved; no truncation",
        "zero_padding": "drop exactly zero noise rows only",
        "split_provenance": "publisher train/ partition; per-house membership not independently checked",
    }
    identities: set[str] = set()
    try:
        with tarfile.open(output.source, mode="r|") as archive:
            for number, member in enumerate(archive, start=1):
                output.check()
                if number > max_archive_members:
                    raise PreparationLimit("Archive member count bound reached")
                path = _safe_member(member.name)
                if not member.isfile() or path.suffix != ".npy" or len(path.parts) != 1:
                    raise ValueError("BRIR shard must contain only flat regular NPY members")
                if output.manifest["converted_count"] >= max_scenes:
                    raise PreparationLimit("Scene count bound reached")
                identity = path.stem
                if (
                    re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity) is None
                    or identity in identities
                ):
                    raise ValueError("Scene IDs must be safe and unique within the shard")
                identities.add(identity)
                with archive.extractfile(member) as handle:
                    payload = _bounded_payload(handle, member.size, MAX_BRIR_MEMBER_BYTES)
                array = _npy_array(payload)
                converted, kind, original_indices = convert_scene_array(array)
                descriptor, temp = tempfile.mkstemp(
                    prefix=".scene-", suffix=".npz", dir=output.root
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        np.savez_compressed(handle, arr_0=converted)
                        handle.flush()
                        os.fsync(handle.fileno())
                    record = output.publish(Path(temp), ".npz")
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temp)
                logicals = [f"brirs/{identity}/{index}.npz" for index in range(len(converted))]
                for index, logical in enumerate(logicals):
                    output.manifest["assets"][logical] = {
                        **record,
                        "index": index,
                        "array_key": "arr_0",
                    }
                output.manifest["scenes"].append(
                    {
                        "id": identity,
                        "noise_type": kind,
                        "source_rir": logicals[0],
                        "noise_rirs": logicals[1:],
                        "source_member": member.name,
                        "source_indices": original_indices,
                        "source_samples": array.shape[-1],
                        "samples": converted.shape[-1],
                    }
                )
                output.manifest["converted_count"] += 1
                output.save()
        if not identities:
            raise ValueError("BRIR shard contains no scenes")
        output.finish()
    except Exception as error:
        output.failure(error)
        raise
    return output.manifest


def prepare_wham_noise(
    source: str | Path,
    expected_sha256: str,
    output_dir: str | Path,
    *,
    max_files: int = 20000,
    max_output_bytes: int = 8 * 1024**3,
    max_seconds: float = 3600,
    external_check: Callable[[], None] | None = None,
) -> dict:
    """Convert only the publisher's training noise from a verified local ZIP."""
    if type(max_files) is not int or not 0 < max_files <= 20000:
        raise ValueError("max_files must be in [1, 20000]")
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("WHAM noise conversion requires soundfile") from error
    output = _Output(
        Path(source),
        expected_sha256,
        Path(output_dir),
        kind="nat_wham_noise",
        max_output_bytes=max_output_bytes,
        max_seconds=max_seconds,
        external_check=external_check,
    )
    output.manifest["provenance"] = {
        "publisher": "http://wham.whisper.ai/",
        "declared_license": "CC-BY-NC-4.0",
        "split": "wham_noise/tr/ only",
        "channel_policy": "first channel, index 0",
        "conversion": "16 kHz float32 stereo WAV -> first channel PCM16; common peak gain if needed",
        "duration": "preserved; renderer performs crop and fade",
    }
    identities: set[str] = set()
    try:
        _bound_zip_directory(output.source)
        with zipfile.ZipFile(output.source) as archive:
            if len(archive.infolist()) > 40000:
                raise PreparationLimit("WHAM ZIP member count bound reached")
            for member in archive.infolist():
                output.check()
                path = _safe_member(member.filename)
                if member.is_dir():
                    continue
                if path.parts[:2] != ("wham_noise", "tr"):
                    continue
                if len(path.parts) != 3 or path.suffix.lower() != ".wav":
                    raise ValueError("WHAM training entries must be flat WAV files")
                if output.manifest["converted_count"] >= max_files:
                    raise PreparationLimit("Noise file count bound reached")
                if path.name in identities:
                    raise ValueError("Duplicate training noise filename")
                identities.add(path.name)
                with archive.open(member) as handle:
                    payload = _bounded_payload(handle, member.file_size, MAX_NOISE_MEMBER_BYTES)
                with sf.SoundFile(io.BytesIO(payload)) as audio:
                    if (
                        audio.samplerate != TARGET_RATE
                        or audio.channels != 2
                        or audio.subtype != "FLOAT"
                        or not 0 < audio.frames <= 60 * TARGET_RATE
                    ):
                        raise ValueError(
                            "WHAM training noise must be 16 kHz stereo float32 WAV <=60s"
                        )
                    mono = audio.read(dtype="float32", always_2d=True)[:, 0]
                if not np.isfinite(mono).all() or not np.any(mono):
                    raise ValueError("Training noise must be finite and nonzero")
                peak = float(np.abs(mono).max())
                gain = min(1.0, (32767 / 32768) / peak)
                pcm = (
                    np.rint(mono.astype(np.float64) * gain * 32768)
                    .clip(-32768, 32767)
                    .astype("<i2")
                )
                # The renderer loads only the first ten seconds and centers
                # them before fading. Validate that exact quantized population,
                # rather than a longer or unquantized source that it never sees.
                prefix = pcm[: 10 * TARGET_RATE]
                if int(prefix.max()) == int(prefix.min()):
                    raise ValueError(
                        "Rendered PCM16 noise prefix must have nonzero centered variance"
                    )
                descriptor, temp = tempfile.mkstemp(
                    prefix=".noise-", suffix=".wav", dir=output.root
                )
                os.close(descriptor)
                try:
                    with wave.open(temp, "wb") as handle:
                        handle.setnchannels(1)
                        handle.setsampwidth(2)
                        handle.setframerate(TARGET_RATE)
                        handle.writeframes(pcm.tobytes())
                    record = output.publish(Path(temp), ".wav")
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temp)
                logical = f"noise/{path.name}"
                output.manifest["assets"][logical] = {
                    **record,
                    "source_member": member.filename,
                    "samples": len(mono),
                    "peak_gain": gain,
                }
                output.manifest["noise_files"].append(logical)
                output.manifest["converted_count"] += 1
                output.save()
        if not identities:
            raise ValueError("WHAM ZIP contains no training noise")
        output.finish()
    except Exception as error:
        output.failure(error)
        raise
    return output.manifest


def _read_component(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = handle.read(64 * 1024**2 + 1)
    if len(payload) > 64 * 1024**2:
        raise ValueError("Component manifest exceeds 64 MiB")
    content = json.loads(payload)
    if (
        not isinstance(content, dict)
        or content.get("schema_version") != 1
        or content.get("status") != "complete"
        or content.get("sample_rate") != TARGET_RATE
        or content.get("split") != "train"
    ):
        raise ValueError("Assembly requires complete schema-1 16 kHz training components")
    source = content.get("source", {})
    if (
        source.get("verified") is not True
        or type(source.get("bytes")) is not int
        or source["bytes"] <= 0
    ):
        raise ValueError("Component requires a verified source receipt")
    _digest(source.get("sha256"))
    if not isinstance(content.get("assets"), dict) or not content["assets"]:
        raise ValueError("Component requires nonempty assets")
    if not isinstance(content.get("provenance"), dict):
        raise ValueError("Component requires explicit source provenance")
    return content


def _write_immutable_json(path: Path, content: dict) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix=".assembled-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(content, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(Path(temp))
        size = Path(temp).stat().st_size
        os.link(temp, path)  # Exclusive atomic publication; no prior file replaced.
        return size, digest
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)


def assemble_nat_manifest(
    brir_manifests: list[str | Path],
    noise_manifest: str | Path,
    output_manifest: str | Path,
    *,
    endpoint: str,
    bucket: str,
    prefix: str,
    upload_plan: str | Path | None = None,
    allow_pilot: bool = False,
    brir_inventory: str | Path | None = None,
    max_seconds: float = 10800,
    max_total_object_bytes: int = 256 * 1024**3,
    _remote_verifier: Callable[[str, str, int], None] | None = None,
) -> dict:
    """Verify local components and publish a remote manifest plus upload plan.

    By default no remote operation occurs. The rolling stager alone supplies
    _remote_verifier: it must verify each key against a durable full-GET receipt
    (including size and SHA-256) and recheck receipts from earlier runs remotely.
    This route preserves all component/source/count validation but does not
    require converted payloads to remain on disk. Full-corpus mode checks all source
    shard receipts against the pinned inventory and requires 70,000 BRIR scenes
    and 20,000 training noise files. Explicit pilot mode is marked in provenance
    and cannot establish complete-corpus readiness.
    """
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Endpoint must be HTTPS without credentials, query or path")
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None:
        raise ValueError("Bucket must be a valid lowercase S3 bucket name")
    if not prefix or str(_safe_member(prefix)) != prefix or prefix in (".", "/"):
        raise ValueError("Prefix must be a canonical nonempty relative object prefix")
    if not 0 < len(brir_manifests) <= 70:
        raise ValueError("Supply one to seventy BRIR component manifests")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if type(max_total_object_bytes) is not int or max_total_object_bytes <= 0:
        raise ValueError("max_total_object_bytes must be positive")
    output_path = Path(output_manifest).expanduser().absolute()
    plan_path = (
        Path(upload_plan).expanduser().absolute()
        if upload_plan is not None
        else output_path.with_name(output_path.name + ".upload.json")
    )
    if (
        output_path == plan_path
        or output_path.exists()
        or output_path.is_symlink()
        or plan_path.exists()
        or plan_path.is_symlink()
    ):
        raise FileExistsError("Manifest and upload plan require distinct new paths")
    deadline = time.monotonic() + max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise PreparationLimit("Manifest assembly runtime bound reached")

    inventory = {}
    if not allow_pilot:
        inventory_path = (
            Path(brir_inventory)
            if brir_inventory is not None
            else (
                Path(__file__).resolve().parents[1]
                / "docs/validation/wavjepa-nat-assets-2026-10-23.json"
            )
        )
        with inventory_path.open("rb") as handle:
            inventory_payload = handle.read(1024**2 + 1)
        if len(inventory_payload) > 1024**2:
            raise ValueError("BRIR inventory exceeds the 1 MiB metadata bound")
        value = json.loads(inventory_payload)["brir"]
        if value["repository"] != BRIR_REPOSITORY or value["revision"] != BRIR_REVISION:
            raise ValueError("BRIR inventory does not match the pinned publisher revision")
        inventory = {item["path"]: item for item in value["shards"]}
        if len(inventory) != 70:
            raise ValueError("Full inventory requires seventy distinct source shards")
    combined = {
        "schema_version": 1,
        "sample_rate": TARGET_RATE,
        "split": "train",
        "bucket": bucket,
        "endpoint": endpoint.rstrip("/"),
        "scenes": [],
        "noise_files": [],
        "assets": {},
        "provenance": {
            "coverage": "pilot" if allow_pilot else "full",
            "remote_verified": _remote_verifier is not None,
            "assembly": (
                "all objects enrolled with full-GET SHA-256 receipts; prior-run receipts rechecked"
                if _remote_verifier is not None
                else "local object sizes and SHA-256 verified; not uploaded"
            ),
            "brir_repository": BRIR_REPOSITORY,
            "brir_revision": BRIR_REVISION,
            "component_manifests": [],
        },
    }
    source_paths: set[str] = set()
    scene_ids: set[str] = set()
    objects: dict[str, dict] = {}
    total_bytes = 0
    component_paths = [Path(path).expanduser().resolve() for path in brir_manifests]
    component_paths.append(Path(noise_manifest).expanduser().resolve())
    if len(set(component_paths)) != len(component_paths):
        raise ValueError("Component manifest paths must be unique")
    for component_index, path in enumerate(component_paths):
        check()
        component = _read_component(path)
        is_noise = component_index == len(component_paths) - 1
        provenance = component["provenance"]
        if component["kind"] != ("nat_wham_noise" if is_noise else "nat_brir_shard"):
            raise ValueError("Wrong component kind for manifest assembly")
        if is_noise:
            if (
                provenance.get("publisher") != "http://wham.whisper.ai/"
                or provenance.get("split") != "wham_noise/tr/ only"
                or provenance.get("channel_policy") != "first channel, index 0"
            ):
                raise ValueError("Noise source must be the WHAM training partition")
            references = component.get("noise_files", [])
            if component.get("scenes") or not isinstance(references, list) or not references:
                raise ValueError("Noise component must contain only noise references")
            if len(set(references)) != len(references):
                raise ValueError("Duplicate noise reference")
            if not allow_pilot and len(references) != 20000:
                raise ValueError("Full corpus requires 20,000 WHAM training files")
            combined["noise_files"].extend(references)
        else:
            source_path = provenance.get("path")
            if (
                provenance.get("repository") != BRIR_REPOSITORY
                or provenance.get("revision") != BRIR_REVISION
                or provenance.get("source_sample_rate") != SOURCE_RATE
                or not isinstance(source_path, str)
                or re.fullmatch(r"train/webdataset_output/shard-0000[0-6][0-9]\.tar", source_path)
                is None
                or source_path in source_paths
            ):
                raise ValueError("BRIR components need distinct pinned training shard provenance")
            source_paths.add(source_path)
            if not allow_pilot and (
                source_path not in inventory
                or inventory[source_path]["sha256"] != component["source"]["sha256"]
                or inventory[source_path]["bytes"] != component["source"]["bytes"]
            ):
                raise ValueError("BRIR source receipt does not match the pinned inventory")
            scenes = component.get("scenes")
            if not isinstance(scenes, list) or not scenes or component.get("noise_files"):
                raise ValueError("BRIR component must contain scenes only")
            if not allow_pilot and len(scenes) != 1000:
                raise ValueError("Full corpus requires 1,000 scenes in each BRIR shard")
            references = []
            for scene in scenes:
                identity = scene.get("id")
                if not isinstance(identity, str) or not identity or identity in scene_ids:
                    raise ValueError("Scene IDs must be globally unique")
                scene_ids.add(identity)
                noise_refs = scene.get("noise_rirs")
                kind = scene.get("noise_type")
                if not isinstance(noise_refs, list) or (
                    len(noise_refs) != 1
                    if kind == "localized"
                    else kind != "diffuse" or len(noise_refs) not in (3, 4, 5)
                ):
                    raise ValueError(
                        "Scene requires one localized or three to five diffuse noise BRIRs"
                    )
                logicals = [scene.get("source_rir"), *noise_refs]
                if len(set(logicals)) != len(logicals) or any(
                    name not in component["assets"] for name in logicals
                ):
                    raise ValueError("Scene references must be distinct and present")
                records = [component["assets"][name] for name in logicals]
                if len({record.get("key") for record in records}) != 1 or [
                    record.get("index") for record in records
                ] != list(range(len(records))):
                    raise ValueError(
                        "Source and noise must refer to matched indices of one packed scene"
                    )
                references.extend(logicals)
                combined["scenes"].append(
                    {key: scene[key] for key in ("id", "noise_type", "source_rir", "noise_rirs")}
                )
        if component.get("converted_count") != (
            len(references) if is_noise else len(component["scenes"])
        ):
            raise ValueError("Component count differs from its references")
        if len(references) != len(set(references)) or set(references) != set(component["assets"]):
            raise ValueError("Component assets must cover exactly its unique references")
        local_keys = {}
        for name, record in component["assets"].items():
            check()
            if (
                not isinstance(name, str)
                or str(_safe_member(name)) != name
                or name in combined["assets"]
            ):
                raise ValueError("Duplicate or unsafe logical asset path")
            digest = _digest(record.get("sha256"))
            suffix = ".wav" if is_noise else ".npz"
            local_key = record.get("key")
            if local_key != f"objects/{digest}{suffix}" or Path(name).suffix != suffix:
                raise ValueError("Asset key must match its content-addressed object and format")
            size = record.get("bytes")
            if type(size) is not int or not 0 < size <= 3 * 1024**2:
                raise ValueError("Converted asset size must fit the bounded scene/noise format")
            if not is_noise and (
                type(record.get("index")) is not int or record.get("array_key") != "arr_0"
            ):
                raise ValueError("BRIR assets require packed integer indices and arr_0")
            if is_noise and record.get("index") is not None:
                raise ValueError("Noise WAV cannot use a packed array index")
            if (
                is_noise
                and record.get("source_member") != f"wham_noise/tr/{PurePosixPath(name).name}"
            ):
                raise ValueError(
                    "Noise object must retain a matching WHAM training source reference"
                )
            if local_key in local_keys and local_keys[local_key] != (size, digest):
                raise ValueError("Repeated object references have inconsistent metadata")
            if local_key not in local_keys:
                physical = (path.parent / local_key).resolve()
                remote_key = f"{prefix}/{local_key}"
                if _remote_verifier is not None:
                    _remote_verifier(remote_key, digest, size)
                elif (
                    not physical.is_relative_to(path.parent)
                    or not physical.is_file()
                    or physical.stat().st_size != size
                    or sha256_file(physical, check) != digest
                ):
                    raise ValueError(
                        "Local converted object failed path, size or SHA-256 verification"
                    )
                local_keys[local_key] = (size, digest)
                if remote_key not in objects:
                    total_bytes += size
                    if total_bytes > max_total_object_bytes:
                        raise PreparationLimit("Assembly object byte bound reached")
                    objects[remote_key] = {
                        "local_path": str(physical) if _remote_verifier is None else None,
                        "key": remote_key,
                        "bytes": size,
                        "sha256": digest,
                    }
            combined["assets"][name] = {**record, "key": f"{prefix}/{local_key}"}
        if component.get("output_bytes") != sum(size for size, _ in local_keys.values()):
            raise ValueError("Component output byte total differs from verified objects")
        combined["provenance"]["component_manifests"].append(
            {
                "sha256": sha256_file(path, check),
                "kind": component["kind"],
                "source": component["source"],
                "provenance": provenance,
            }
        )
    if not allow_pilot and (len(source_paths) != 70 or len(scene_ids) != 70000):
        raise ValueError(
            "Full corpus requires all seventy distinct training shards and 70,000 scenes"
        )
    if {scene["noise_type"] for scene in combined["scenes"]} != {"localized", "diffuse"}:
        raise ValueError("Manifest must contain both localized and diffuse scenes")
    check()
    size, digest = _write_immutable_json(output_path, combined)
    plan = {
        "schema_version": 1,
        "status": (
            "remote_objects_verified_manifest_not_published"
            if _remote_verifier is not None
            else "offline_verified_not_uploaded"
        ),
        "bucket": bucket,
        "endpoint": endpoint.rstrip("/"),
        "prefix": prefix,
        "manifest": {
            "local_path": str(output_path),
            "key": f"{prefix}/manifest.json",
            "bytes": size,
            "sha256": digest,
        },
        "objects": list(objects.values()),
        "total_object_bytes": total_bytes,
        "publication_order": "Upload and readback-verify all objects before publishing manifest.json",
    }
    _write_immutable_json(plan_path, plan)
    return {
        "manifest": str(output_path),
        "manifest_sha256": digest,
        "upload_plan": str(plan_path),
        "coverage": combined["provenance"]["coverage"],
        "scene_count": len(scene_ids),
        "noise_count": len(combined["noise_files"]),
        "object_bytes": total_bytes,
        "status": (
            "remote_objects_verified_manifest_not_published"
            if _remote_verifier is not None
            else "offline_verified_not_uploaded"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("brir", "wham", "assemble"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output-dir", type=Path, help="New directory; never overwritten")
    parser.add_argument(
        "--source-member", help="BRIR publisher path train/webdataset_output/shard-NNNNNN.tar"
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-output-bytes", type=int)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--brir-manifest", type=Path, action="extend", nargs="+", default=[])
    parser.add_argument("--noise-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--upload-plan", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--bucket")
    parser.add_argument("--prefix")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--brir-inventory", type=Path)
    args = parser.parse_args()
    kwargs = {"max_seconds": args.max_seconds}
    if args.max_output_bytes is not None:
        kwargs["max_output_bytes"] = args.max_output_bytes
    try:
        if args.kind == "assemble":
            if not args.brir_manifest or any(
                value is None
                for value in (
                    args.noise_manifest,
                    args.output_manifest,
                    args.endpoint,
                    args.bucket,
                    args.prefix,
                )
            ):
                parser.error(
                    "assemble requires BRIR/noise manifests, output manifest, endpoint, bucket and prefix"
                )
            result = assemble_nat_manifest(
                args.brir_manifest,
                args.noise_manifest,
                args.output_manifest,
                endpoint=args.endpoint,
                bucket=args.bucket,
                prefix=args.prefix,
                upload_plan=args.upload_plan,
                allow_pilot=args.allow_pilot,
                brir_inventory=args.brir_inventory,
                max_seconds=args.max_seconds,
                **(
                    {"max_total_object_bytes": args.max_output_bytes}
                    if args.max_output_bytes is not None
                    else {}
                ),
            )
            print(json.dumps(result))
            return
        if args.source is None or args.expected_sha256 is None or args.output_dir is None:
            parser.error("conversion requires --source, --expected-sha256 and --output-dir")
        if args.kind == "brir":
            if args.source_member is None:
                parser.error("--source-member is required for BRIR conversion")
            if args.max_items is not None:
                kwargs["max_scenes"] = args.max_items
            result = prepare_brir_shard(
                args.source,
                args.expected_sha256,
                args.output_dir,
                source_member=args.source_member,
                **kwargs,
            )
        else:
            if args.source_member is not None:
                parser.error("--source-member is only valid for BRIR conversion")
            if args.max_items is not None:
                kwargs["max_files"] = args.max_items
            result = prepare_wham_noise(
                args.source, args.expected_sha256, args.output_dir, **kwargs
            )
    except (ValueError, RuntimeError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Nat preparation stopped: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "converted_count", "output_bytes")}))


if __name__ == "__main__":
    main()

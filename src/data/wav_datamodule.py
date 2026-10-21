"""Explicit local WAV manifests without downloads, resampling, or auto-splits.

Every nonempty JSONL line must be an object with a ``path`` field. Relative paths
are resolved against that manifest's directory. Files must be uncompressed
16-bit PCM mono WAV at the configured sample rate. Inputs are cropped from the
start or zero-padded to ``num_samples``. Users remain responsible for splitting
by recording, speaker, subject, or other appropriate independent unit: distinct
file paths alone cannot establish statistically independent splits.
"""

from __future__ import annotations

import array
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class _ManifestEntry:
    path: Path
    identity: tuple[int, int]
    location: str
    group_id: str | None = None
    recording_id: str | None = None


def _manifest_entries(manifest: str | Path) -> list[_ManifestEntry]:
    manifest = Path(manifest).expanduser().resolve()
    entries: list[_ManifestEntry] = []
    seen: set[tuple[int, int]] = set()
    try:
        with manifest.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                location = f"{manifest}:{number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{location}: invalid JSON") from error
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("path"), str)
                    or not record["path"].strip()
                ):
                    raise ValueError(f"{location}: expected an object with a nonempty string path")
                identifiers = {}
                for field in ("group_id", "recording_id"):
                    if field in record:
                        value = record[field]
                        if not isinstance(value, str) or not value.strip():
                            raise ValueError(f"{location}: {field} must be a nonempty string")
                        identifiers[field] = value.strip()
                path = Path(record["path"]).expanduser()
                if not path.is_absolute():
                    path = manifest.parent / path
                try:
                    path = path.resolve(strict=True)
                    stat = path.stat()
                    if not path.is_file():
                        raise ValueError(f"{location}: WAV path is not a file: {path}")
                except OSError as error:
                    raise ValueError(
                        f"{location}: cannot access WAV file {path}: {error}"
                    ) from error
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    raise ValueError(f"Duplicate physical WAV file in {location}: {path}")
                seen.add(identity)
                entries.append(_ManifestEntry(path, identity, location, **identifiers))
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read UTF-8 JSONL manifest {manifest}: {error}") from error
    if not entries:
        raise ValueError(f"Manifest must contain at least one WAV path: {manifest}")
    return entries


def _manifest_paths(manifest: str | Path) -> list[Path]:
    """Backward-compatible path reader, including physical-file duplicate checks."""
    return [entry.path for entry in _manifest_entries(manifest)]


def _split_entries(manifests: dict[str, str]) -> dict[str, list[_ManifestEntry]]:
    entries = {split: _manifest_entries(path) for split, path in manifests.items()}
    seen_files: dict[tuple[int, int], str] = {}
    for split, records in entries.items():
        for record in records:
            if record.identity in seen_files:
                raise ValueError(
                    f"Physical WAV files overlap across splits ({seen_files[record.identity]}, "
                    f"{split}): {record.path} ({record.location})"
                )
            seen_files[record.identity] = split
    for field in ("group_id", "recording_id"):
        if not any(
            getattr(record, field) is not None for rows in entries.values() for record in rows
        ):
            continue
        seen_groups: dict[str, str] = {}
        for split, records in entries.items():
            for record in records:
                value = getattr(record, field)
                if value is None:
                    raise ValueError(
                        f"{record.location}: {field} is required on every record when provided "
                        "on any record; partial metadata cannot establish split disjointness"
                    )
                if value in seen_groups and seen_groups[value] != split:
                    raise ValueError(
                        f"{field} overlap across splits ({seen_groups[value]}, {split}): "
                        f"{value!r} ({record.location})"
                    )
                seen_groups[value] = split
    return entries


def _positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_wav(handle: wave.Wave_read, path: Path, sample_rate: int) -> None:
    if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
        raise ValueError(f"Expected uncompressed 16-bit PCM mono WAV: {path}")
    if handle.getframerate() != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {handle.getframerate()}: {path}")
    if handle.getnframes() <= 0:
        raise ValueError(f"WAV must contain at least one sample: {path}")


def _inspect_wav(path: Path, sample_rate: int) -> int:
    """Check the declared payload's end with a seek, without decoding the full file."""
    try:
        with wave.open(str(path), "rb") as handle:
            _validate_wav(handle, path, sample_rate)
            frames = handle.getnframes()
            handle.setpos(frames - 1)
            # Asking for one extra frame also detects a trailing incomplete PCM sample.
            if len(handle.readframes(2)) != 2:
                raise ValueError(f"Truncated or misaligned PCM sample data: {path}")
            return frames
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Cannot read valid PCM WAV {path}: {error}") from error


class WAVManifestDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, paths: list[Path], sample_rate: int = 16_000, num_samples: int = 1024
    ) -> None:
        if not paths:
            raise ValueError("WAV dataset must not be empty")
        _positive_integer(sample_rate, "sample_rate")
        _positive_integer(num_samples, "num_samples")
        self.paths = [Path(path) for path in paths]
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.frame_counts = [_inspect_wav(path, sample_rate) for path in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        try:
            with wave.open(str(path), "rb") as handle:
                _validate_wav(handle, path, self.sample_rate)
                requested = min(self.num_samples, handle.getnframes())
                raw = handle.readframes(requested)
        except (OSError, EOFError, wave.Error) as error:
            raise ValueError(f"Cannot read valid PCM WAV {path}: {error}") from error
        if len(raw) != requested * 2:
            raise ValueError(f"Truncated PCM sample data: {path}")
        values = array.array("h")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        waveform = torch.zeros(self.num_samples, dtype=torch.float32)
        waveform[: len(values)] = torch.tensor(values, dtype=torch.float32) / 32768.0
        return {"waveform": waveform}


class WAVManifestDataModule(L.LightningDataModule):
    """Load physically disjoint JSONL splits; validate supplied group/recording IDs.

    These checks do not identify copied audio, overlapping crops, or missing group
    metadata. Independent splitting remains the dataset curator's responsibility.
    Keep manifests and audio immutable during a run: setup validates a snapshot.
    """

    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        sample_rate: int = 16_000,
        num_samples: int = 1024,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("batch_size", batch_size),
            ("sample_rate", sample_rate),
            ("num_samples", num_samples),
        ):
            _positive_integer(value, name)
        _positive_integer(num_workers, "num_workers", minimum=0)
        if not isinstance(pin_memory, bool) or not isinstance(persistent_workers, bool):
            raise ValueError("pin_memory and persistent_workers must be booleans")
        if prefetch_factor is not None:
            _positive_integer(prefetch_factor, "prefetch_factor")
        if num_workers == 0 and (persistent_workers or prefetch_factor is not None):
            raise ValueError("persistent_workers and prefetch_factor require num_workers > 0")
        self.save_hyperparameters(logger=False)
        self.data_train: WAVManifestDataset | None = None
        self.data_val: WAVManifestDataset | None = None
        self.data_test: WAVManifestDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        entries = _split_entries(
            {split: self.hparams[f"{split}_manifest"] for split in ("train", "val", "test")}
        )
        datasets = {
            split: WAVManifestDataset(
                [record.path for record in rows], self.hparams.sample_rate, self.hparams.num_samples
            )
            for split, rows in entries.items()
        }
        # Assign only after every split validates, avoiding a half-initialized module.
        for split, dataset in datasets.items():
            setattr(
                self,
                f"data_{split}",
                dataset,
            )

    def _loader(self, dataset: WAVManifestDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)


def validate_manifests(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    sample_rate: int = 16_000,
    num_samples: int = 1024,
) -> dict:
    """Read-only preflight with bounded reads per WAV; never hashes the audio corpus."""
    _positive_integer(sample_rate, "sample_rate")
    _positive_integer(num_samples, "num_samples")
    entries = _split_entries({"train": train_manifest, "val": val_manifest, "test": test_manifest})
    summaries = {}
    for split, rows in entries.items():
        dataset = WAVManifestDataset([row.path for row in rows], sample_rate, num_samples)
        summaries[split] = {
            "files": len(dataset),
            "source_frames": sum(dataset.frame_counts),
            "cropped_files": sum(frames > num_samples for frames in dataset.frame_counts),
            "padded_files": sum(frames < num_samples for frames in dataset.frame_counts),
        }
    checked_ids = [
        field
        for field in ("group_id", "recording_id")
        if getattr(entries["train"][0], field) is not None
    ]
    return {
        "valid": True,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "splits": summaries,
        "checks": {
            "physical_files_disjoint": True,
            "group_fields_disjoint": checked_ids,
            "pcm_headers_and_payload_extents": True,
            "content_hashes": False,
        },
        "limitations": [
            "Distinct physical files may contain copied audio or overlapping crops.",
            "Group guarantees depend on complete, correct user-provided identifiers.",
            "Validation is a snapshot; keep manifests and audio immutable during training.",
        ],
    }

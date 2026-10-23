"""Explicit local WAV manifests without downloads, resampling, or auto-splits.

Every nonempty JSONL line has a ``path`` and optional ``offset_samples`` (default
zero), ``segment_samples`` (default the rest of the recording), ``group_id`` and
``recording_id``. Offsets/lengths use samples at the configured sample rate, not
seconds. Relative paths are resolved against the manifest's directory. Files
must be uncompressed 16-bit PCM mono WAV at the configured sample rate. Training
can sample a random contiguous crop inside the segment; evaluation always starts
at its offset. ``require_full_length`` forbids padding future targets.
Users remain responsible for splitting
by recording, speaker, subject, or other appropriate independent unit: distinct
file paths alone cannot establish statistically independent splits.
"""

from __future__ import annotations

import array
import hashlib
import json
import sys
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    Sampler,
)
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class _ManifestEntry:
    path: Path
    identity: tuple[int, int]
    location: str
    group_id: str | None = None
    recording_id: str | None = None
    offset_samples: int = 0
    segment_samples: int | None = None


def _manifest_entries(manifest: str | Path) -> list[_ManifestEntry]:
    manifest = Path(manifest).expanduser().resolve()
    entries: list[_ManifestEntry] = []
    seen: set[tuple[tuple[int, int], int, int | None]] = set()
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
                unknown = set(record) - {
                    "path",
                    "group_id",
                    "recording_id",
                    "offset_samples",
                    "segment_samples",
                }
                if unknown:
                    raise ValueError(f"{location}: unknown manifest fields: {sorted(unknown)}")
                offset = record.get("offset_samples", 0)
                _positive_integer(offset, f"{location}: offset_samples", minimum=0)
                segment = record.get("segment_samples")
                if "segment_samples" in record:
                    _positive_integer(segment, f"{location}: segment_samples")
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
                segment_identity = (identity, offset, segment)
                if segment_identity in seen:
                    raise ValueError(f"Duplicate physical WAV segment in {location}: {path}")
                seen.add(segment_identity)
                entries.append(
                    _ManifestEntry(
                        path,
                        identity,
                        location,
                        **identifiers,
                        offset_samples=offset,
                        segment_samples=segment,
                    )
                )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read UTF-8 JSONL manifest {manifest}: {error}") from error
    if not entries:
        raise ValueError(f"Manifest must contain at least one WAV path: {manifest}")
    return entries


def _manifest_paths(manifest: str | Path) -> list[Path]:
    """Read paths after validating the manifest's segment and identifier schema."""
    return [entry.path for entry in _manifest_entries(manifest)]


def _split_entries(manifests: dict[str, str]) -> dict[str, list[_ManifestEntry]]:
    entries = {split: _manifest_entries(path) for split, path in manifests.items()}
    seen_files: dict[tuple[int, int], str] = {}
    for split, records in entries.items():
        for record in records:
            if record.identity in seen_files and seen_files[record.identity] != split:
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
        self,
        paths: list[Path] | list[_ManifestEntry],
        sample_rate: int = 16_000,
        num_samples: int = 1024,
        require_full_length: bool = False,
        random_crop: bool = False,
        seed: int = 0,
    ) -> None:
        if not paths:
            raise ValueError("WAV dataset must not be empty")
        _positive_integer(sample_rate, "sample_rate")
        _positive_integer(num_samples, "num_samples")
        if not isinstance(require_full_length, bool):
            raise ValueError("require_full_length must be a boolean")
        if not isinstance(random_crop, bool):
            raise ValueError("random_crop must be a boolean")
        _positive_integer(seed, "seed", minimum=0)
        self.paths = [
            path.path if isinstance(path, _ManifestEntry) else Path(path) for path in paths
        ]
        self.offsets = [
            path.offset_samples if isinstance(path, _ManifestEntry) else 0 for path in paths
        ]
        self.segment_sizes = [
            path.segment_samples if isinstance(path, _ManifestEntry) else None for path in paths
        ]
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.require_full_length = require_full_length
        self.random_crop = random_crop
        self.seed = seed
        # Multiple segments of one recording share a header/payload inspection.
        inspected = {path: _inspect_wav(path, sample_rate) for path in dict.fromkeys(self.paths)}
        self.frame_counts = [inspected[path] for path in self.paths]
        self.segment_frame_counts = [
            self._segment_length(index, frames) for index, frames in enumerate(self.frame_counts)
        ]
        seen_segments = set()
        for entry, frames in zip(paths, self.segment_frame_counts, strict=True):
            if isinstance(entry, _ManifestEntry):
                key = (entry.identity, entry.offset_samples, frames)
                if key in seen_segments:
                    raise ValueError(
                        f"Duplicate physical WAV segment in {entry.location}: {entry.path}"
                    )
                seen_segments.add(key)
        for path, frames in zip(self.paths, self.segment_frame_counts, strict=True):
            self._require_length(path, frames)

    def _segment_length(self, index: int, frames: int) -> int:
        offset, length = self.offsets[index], self.segment_sizes[index]
        if offset >= frames:
            raise ValueError(
                f"offset_samples={offset} is outside WAV with {frames} samples: {self.paths[index]}"
            )
        if length is None:
            return frames - offset
        if offset + length > frames:
            raise ValueError(
                f"WAV segment ends at {offset + length}, beyond {frames} samples: {self.paths[index]}"
            )
        return length

    def _require_length(self, path: Path, frames: int) -> None:
        if self.require_full_length and frames < self.num_samples:
            raise ValueError(
                f"Forecasting requires at least {self.num_samples} real samples "
                f"({self.num_samples / self.sample_rate:g} seconds), got {frames}: {path}; "
                "padding cannot be used as future audio"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, torch.Tensor]:
        # The sampler carries epoch into worker processes, including persistent
        # workers. Stateless randomness makes crops independent of rank/worker
        # scheduling and reproducible after a checkpoint resumes at an epoch.
        index, epoch = index if isinstance(index, tuple) else (index, 0)
        path = self.paths[index]
        try:
            with wave.open(str(path), "rb") as handle:
                _validate_wav(handle, path, self.sample_rate)
                frames = self._segment_length(index, handle.getnframes())
                self._require_length(path, frames)
                offset = self.offsets[index]
                slack = max(0, frames - self.num_samples)
                if self.random_crop and slack:
                    digest = hashlib.blake2b(
                        f"{self.seed}:{epoch}:{index}".encode("ascii"), digest_size=8
                    ).digest()
                    offset += int.from_bytes(digest, "little") % (slack + 1)
                handle.setpos(offset)
                requested = min(self.num_samples, frames)
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


class _EpochDistributedSampler(DistributedSampler):
    """Shard equally and carry the epoch with each index for stateless cropping.

    Standard DistributedSampler padding repeats at most world_size-1 entries to
    make all ranks execute the same number of steps. Evaluate with one device
    when every validation/test entry must contribute exactly once.
    """

    def __iter__(self) -> Iterator[tuple[int, int]]:
        epoch = self.epoch
        return ((index, epoch) for index in super().__iter__())


class _MinimumBatchSampler(BatchSampler):
    """Merge an undersized final batch into its predecessor without losing rows.

    This preserves the underlying sampler's order, including shuffled training
    order. The final batch may exceed ``batch_size`` by ``minimum_batch_size - 1``.
    Unlike ``drop_last``, every validation/test example is retained.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, minimum_batch_size: int) -> None:
        _positive_integer(minimum_batch_size, "minimum_batch_size")
        if batch_size < minimum_batch_size or len(sampler) < minimum_batch_size:
            raise ValueError("batch size and dataset size must be >= minimum_batch_size")
        super().__init__(sampler, batch_size, drop_last=False)
        self.minimum_batch_size = minimum_batch_size

    def __iter__(self) -> Iterator[list[int]]:
        pending = None
        for batch in super().__iter__():
            if pending is None:
                pending = batch
            elif len(batch) < self.minimum_batch_size:
                pending.extend(batch)
            else:
                yield pending
                pending = batch
        if pending is not None:
            yield pending

    def __len__(self) -> int:
        batches = super().__len__()
        remainder = len(self.sampler) % self.batch_size
        return batches - int(batches > 1 and 0 < remainder < self.minimum_batch_size)


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
        require_full_length: bool = False,
        minimum_batch_size: int = 1,
        train_random_crop: bool = False,
        train_drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("batch_size", batch_size),
            ("sample_rate", sample_rate),
            ("num_samples", num_samples),
            ("minimum_batch_size", minimum_batch_size),
        ):
            _positive_integer(value, name)
        _positive_integer(num_workers, "num_workers", minimum=0)
        _positive_integer(seed, "seed", minimum=0)
        if batch_size < minimum_batch_size:
            raise ValueError("batch_size must be >= minimum_batch_size")
        if not isinstance(pin_memory, bool) or not isinstance(persistent_workers, bool):
            raise ValueError("pin_memory and persistent_workers must be booleans")
        if not isinstance(require_full_length, bool):
            raise ValueError("require_full_length must be a boolean")
        if not isinstance(train_random_crop, bool) or not isinstance(train_drop_last, bool):
            raise ValueError("train_random_crop and train_drop_last must be booleans")
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
                rows,
                self.hparams.sample_rate,
                self.hparams.num_samples,
                require_full_length=self.hparams.require_full_length,
                random_crop=split == "train" and self.hparams.train_random_crop,
                seed=self.hparams.seed,
            )
            for split, rows in entries.items()
        }
        world_size = self.trainer.world_size if self.trainer is not None else 1
        for split, dataset in datasets.items():
            _validate_split_size(
                split,
                len(dataset),
                self.hparams.batch_size,
                self.hparams.minimum_batch_size,
                world_size,
                self.hparams.train_drop_last,
            )
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
        options = {
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "persistent_workers": self.hparams.persistent_workers,
            "prefetch_factor": self.hparams.prefetch_factor,
        }
        if self.hparams.num_workers:
            # CUDA/NCCL may already be initialized; inherited fork state is unsafe.
            # Epochs travel in sampler indices, so no cross-context shared locks are needed.
            options["multiprocessing_context"] = "spawn"
        world_size = self.trainer.world_size if self.trainer is not None else 1
        rank = self.trainer.global_rank if self.trainer is not None else 0
        drop_last = shuffle and self.hparams.train_drop_last
        _validate_split_size(
            "train" if shuffle else "evaluation",
            len(dataset),
            self.hparams.batch_size,
            self.hparams.minimum_batch_size,
            world_size,
            drop_last,
        )
        sampler = _EpochDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=self.hparams.seed,
            drop_last=drop_last,
        )
        if drop_last:
            batch_sampler = BatchSampler(sampler, self.hparams.batch_size, drop_last=True)
        else:
            batch_sampler = _MinimumBatchSampler(
                sampler, self.hparams.batch_size, self.hparams.minimum_batch_size
            )
        # Lightning must not replace this sampler: set Trainer(use_distributed_sampler=False).
        # Its normal epoch hook still calls batch_sampler.sampler.set_epoch().
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            generator=torch.Generator().manual_seed(self.hparams.seed + rank),
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)


def _validate_split_size(
    split: str,
    size: int,
    batch_size: int,
    minimum_batch_size: int,
    world_size: int,
    train_drop_last: bool,
) -> None:
    local_minimum = batch_size if split == "train" and train_drop_last else minimum_batch_size
    required = world_size * local_minimum
    if size < required:
        raise ValueError(
            f"{split} split has {size} recordings/segments; minimum_batch_size="
            f"{minimum_batch_size}, world_size={world_size}, train_drop_last={train_drop_last} "
            f"requires at least {required} recordings/segments"
        )


def validate_manifests(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    sample_rate: int = 16_000,
    num_samples: int = 1024,
    require_full_length: bool = False,
    batch_size: int = 4,
    minimum_batch_size: int = 1,
    world_size: int = 1,
    train_drop_last: bool = False,
) -> dict:
    """Read-only preflight with bounded reads per WAV; never hashes the audio corpus."""
    _positive_integer(sample_rate, "sample_rate")
    _positive_integer(num_samples, "num_samples")
    for name, value in (
        ("batch_size", batch_size),
        ("minimum_batch_size", minimum_batch_size),
        ("world_size", world_size),
    ):
        _positive_integer(value, name)
    if batch_size < minimum_batch_size:
        raise ValueError("batch_size must be >= minimum_batch_size")
    if not isinstance(train_drop_last, bool):
        raise ValueError("train_drop_last must be a boolean")
    entries = _split_entries({"train": train_manifest, "val": val_manifest, "test": test_manifest})
    summaries = {}
    for split, rows in entries.items():
        dataset = WAVManifestDataset(rows, sample_rate, num_samples, require_full_length)
        _validate_split_size(
            split, len(dataset), batch_size, minimum_batch_size, world_size, train_drop_last
        )
        summaries[split] = {
            "files": len(dataset),
            "source_frames": sum(dataset.segment_frame_counts),
            "cropped_files": sum(frames > num_samples for frames in dataset.segment_frame_counts),
            "padded_files": sum(frames < num_samples for frames in dataset.segment_frame_counts),
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
        "require_full_length": require_full_length,
        "batch_size": batch_size,
        "minimum_batch_size": minimum_batch_size,
        "world_size": world_size,
        "train_drop_last": train_drop_last,
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
            "The files count is manifest entries; repeated physical files can define distinct segments.",
            "Multi-device evaluation pads ranks; use one device for exact once-per-entry metrics.",
        ],
    }

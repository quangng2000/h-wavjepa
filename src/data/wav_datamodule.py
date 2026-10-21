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
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


def _manifest_paths(manifest: str | Path) -> list[Path]:
    manifest = Path(manifest).expanduser().resolve(strict=True)
    paths: list[Path] = []
    seen: set[Path] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{manifest}:{number}: invalid JSON") from error
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not record["path"]
        ):
            raise ValueError(f"{manifest}:{number}: expected an object with a nonempty string path")
        path = Path(record["path"]).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{manifest}:{number}: WAV path is not a file: {path}")
        if path in seen:
            raise ValueError(f"Duplicate resolved WAV path in {manifest}: {path}")
        seen.add(path)
        paths.append(path)
    if not paths:
        raise ValueError(f"Manifest must contain at least one WAV path: {manifest}")
    return paths


def _validate_wav(handle: wave.Wave_read, path: Path, sample_rate: int) -> None:
    if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
        raise ValueError(f"Expected uncompressed 16-bit PCM mono WAV: {path}")
    if handle.getframerate() != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {handle.getframerate()}: {path}")
    if handle.getnframes() <= 0:
        raise ValueError(f"WAV must contain at least one sample: {path}")


class WAVManifestDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, paths: list[Path], sample_rate: int = 16_000, num_samples: int = 1024
    ) -> None:
        if not paths:
            raise ValueError("WAV dataset must not be empty")
        if sample_rate <= 0 or num_samples <= 0:
            raise ValueError("sample_rate and num_samples must be positive")
        self.paths = paths
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        for path in paths:
            with wave.open(str(path), "rb") as handle:
                _validate_wav(handle, path, sample_rate)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        with wave.open(str(path), "rb") as handle:
            _validate_wav(handle, path, self.sample_rate)
            requested = min(self.num_samples, handle.getnframes())
            raw = handle.readframes(requested)
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
    """Load three user-provided, file-disjoint JSONL manifests."""

    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        sample_rate: int = 16_000,
        num_samples: int = 1024,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be nonnegative")
        self.save_hyperparameters(logger=False)
        self.data_train: WAVManifestDataset | None = None
        self.data_val: WAVManifestDataset | None = None
        self.data_test: WAVManifestDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        paths = {
            split: _manifest_paths(self.hparams[f"{split}_manifest"])
            for split in ("train", "val", "test")
        }
        seen: set[Path] = set()
        for split, entries in paths.items():
            overlap = seen.intersection(entries)
            if overlap:
                raise ValueError(
                    f"Resolved WAV paths overlap across splits ({split}): {sorted(overlap)}"
                )
            seen.update(entries)
        for split, entries in paths.items():
            setattr(
                self,
                f"data_{split}",
                WAVManifestDataset(entries, self.hparams.sample_rate, self.hparams.num_samples),
            )

    def _loader(self, dataset: WAVManifestDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)

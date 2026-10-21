"""Deterministic paired audio fixtures for testing the training pipeline.

Each adjacent pair contains exactly the same two tone events in opposite order.
Amplitudes, frequencies, phases and event envelopes are identical within a pair,
so none of those nuisance factors predicts its balanced binary order label.
This deliberately simple fixture is not evidence of audio generalization.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


def _stream_seed(seed: int, split: str, pair: int = 0) -> int:
    """Use stable, named streams rather than Python's randomized string hash."""
    message = f"h-wavjepa-synthetic:{seed}:{split}:{pair}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**63)


class SyntheticAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Index ``2k`` is low/high; index ``2k+1`` is high/low for the same events."""

    def __init__(
        self,
        size: int,
        sample_rate: int = 16_000,
        num_samples: int = 1024,
        seed: int = 42,
        split: Literal["train", "val", "test"] = "train",
    ) -> None:
        if size <= 0 or size % 2:
            raise ValueError(
                "Synthetic split size must be positive and even for exact label balance"
            )
        if num_samples < 4 or num_samples % 2:
            raise ValueError("num_samples must be even and at least 4 for two equal-length events")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.size = size
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.seed = seed
        self.split = split

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        generator = torch.Generator().manual_seed(_stream_seed(self.seed, self.split, index // 2))
        random = torch.rand(5, generator=generator)
        amplitude = 0.2 + 0.6 * random[0]
        # Both frequencies are strictly below Nyquist for every sample rate.
        low_frequency = 0.015 + 0.055 * random[1]
        high_frequency = 0.12 + 0.08 * random[2]
        length = self.num_samples // 2
        samples = torch.arange(length, dtype=torch.float32)
        envelope = torch.sin(math.pi * (samples + 0.5) / length).square()
        low = amplitude * envelope * torch.sin(2 * math.pi * (low_frequency * samples + random[3]))
        high = (
            amplitude * envelope * torch.sin(2 * math.pi * (high_frequency * samples + random[4]))
        )
        label = index % 2
        waveform = torch.cat((low, high) if label == 0 else (high, low))
        return {
            "waveform": waveform,
            "order_label": torch.tensor(label, dtype=torch.long),
            # Fixed-size nuisance descriptor, shared by both orders. Frequencies
            # are cycles/sample, not Hz; this is an optional diagnostic target.
            "detail_target": torch.stack((amplitude, low_frequency, high_frequency)),
        }


class SyntheticAudioDataModule(L.LightningDataModule):
    """Balanced fixtures with independently seeded train/validation/test streams."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        num_samples: int = 1024,
        batch_size: int = 4,
        num_workers: int = 0,
        train_size: int = 32,
        val_size: int = 16,
        test_size: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers must be nonnegative")
        self.save_hyperparameters(logger=False)
        self.data_train: SyntheticAudioDataset | None = None
        self.data_val: SyntheticAudioDataset | None = None
        self.data_test: SyntheticAudioDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # Setting up all splits is cheap and keeps each stream independent of
        # the order in which Lightning invokes fit/validate/test.
        for split, size in (
            ("train", self.hparams.train_size),
            ("val", self.hparams.val_size),
            ("test", self.hparams.test_size),
        ):
            setattr(
                self,
                f"data_{split}",
                SyntheticAudioDataset(
                    size=size,
                    sample_rate=self.hparams.sample_rate,
                    num_samples=self.hparams.num_samples,
                    seed=self.hparams.seed,
                    split=split,
                ),
            )

    def _loader(self, dataset: SyntheticAudioDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call setup() before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(_stream_seed(self.hparams.seed, "loader")),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)

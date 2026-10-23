"""Small stdout progress records for bounded production training jobs."""

from __future__ import annotations

import json
import math
import time
from numbers import Real

import torch
from lightning.pytorch.callbacks import Callback


class TrainingProgress(Callback):
    """Report rank-zero optimizer progress without storing tensors or graphs.

    Rates include time spent waiting for inputs, validation, and checkpoints.
    The measurement window restarts with each fit, including checkpoint resume.
    """

    def __init__(
        self, every_n_steps: int = 100, loss_key: str = "train/loss", all_ranks: bool = False
    ):
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if not isinstance(loss_key, str) or not loss_key:
            raise ValueError("loss_key must be a nonempty string")
        if type(all_ranks) is not bool:
            raise ValueError("all_ranks must be a boolean")
        self.all_ranks = all_ranks
        self.every_n_steps, self.loss_key = every_n_steps, loss_key
        self._started_at = None
        self._last_at = None
        self._last_step = None

    def on_train_start(self, trainer, pl_module):
        self._started_at = self._last_at = time.monotonic()
        self._last_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (not self.all_ranks and not trainer.is_global_zero) or self._started_at is None:
            return
        step = int(trainer.global_step)
        updates = step - self._last_step
        if updates < self.every_n_steps:
            return
        now = time.monotonic()
        interval = now - self._last_at
        if interval <= 0:
            return
        value = trainer.callback_metrics.get(self.loss_key)
        if isinstance(value, torch.Tensor):
            value = value.detach().item() if value.numel() == 1 else None
        loss = (
            float(value)
            if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
            else None
        )
        rank_fields = (
            {"rank": int(trainer.global_rank), "world_size": int(trainer.world_size)}
            if self.all_ranks
            else {}
        )
        print(
            json.dumps(
                {
                    "event": "training_progress"
                    if trainer.is_global_zero
                    else "training_rank_progress",
                    **rank_fields,
                    "global_step": step,
                    "elapsed_seconds": now - self._started_at,
                    "interval_seconds": interval,
                    "interval_updates": updates,
                    "updates_per_second": updates / interval,
                    "loss": loss,
                },
                allow_nan=False,
            ),
            flush=True,
        )
        self._last_at, self._last_step = now, step

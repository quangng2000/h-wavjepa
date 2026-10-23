"""Step-based latest checkpoints with an explicit final save.

Lightning owns checkpoint serialization, optimizer/loop state and retention.
This adapter publishes local files atomically and saves at normal train end.
It does not upload files or checkpoint iterable-loader positions.
"""

import os
import uuid
from pathlib import Path
from weakref import proxy

from lightning.pytorch.callbacks import ModelCheckpoint


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StepCheckpoint(ModelCheckpoint):
    """Keep one latest full checkpoint, independent of validation improvement.

    ``last.ckpt`` is a local symlink by default, avoiding a second large optimizer
    checkpoint. Durable exporters should upload the regular step file or follow
    the symlink. A separate optional ModelCheckpoint can retain the best model.
    """

    def __init__(self, every_n_train_steps: int = 5000, **kwargs):
        options = {
            "monitor": None,
            "save_top_k": 1,
            "save_last": "link",
            "save_weights_only": False,
            "every_n_epochs": 0,
            "save_on_train_epoch_end": False,
        } | kwargs
        if type(every_n_train_steps) is not int or every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be a positive integer")
        if options["monitor"] is not None or options["save_top_k"] != 1:
            raise ValueError("StepCheckpoint requires monitor=null and save_top_k=1")
        if options["save_weights_only"] or options["save_last"] not in (True, "link"):
            raise ValueError("StepCheckpoint requires full state and save_last=true or link")
        if options["every_n_epochs"] != 0 or options.get("train_time_interval") is not None:
            raise ValueError(
                "StepCheckpoint uses steps; epoch/time checkpoint triggers must be disabled"
            )
        super().__init__(every_n_train_steps=every_n_train_steps, **options)
        self._pending_removals = []

    def _save_checkpoint(self, trainer, filepath):
        """Serialize to a sibling file; publish only flushed, complete bytes.

        All ranks still participate in Lightning serialization and barriers.
        This callback supports local filesystem paths, including mounted disks.
        """
        if "://" in str(filepath):
            raise ValueError(
                "StepCheckpoint requires a local path; export durable copies separately"
            )
        destination = Path(filepath)
        temporary = trainer.strategy.broadcast(
            str(destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial"))
            if trainer.is_global_zero
            else None
        )
        temporary = Path(temporary)
        try:
            trainer.save_checkpoint(temporary, self.save_weights_only)
            outcome = None
            if trainer.is_global_zero:
                try:
                    with temporary.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    _sync_directory(destination.parent)
                    outcome = {"ok": True}
                except Exception as error:
                    outcome = {"error": type(error).__name__}
            outcome = trainer.strategy.broadcast(outcome)
            if "error" in outcome:
                raise RuntimeError(f"Atomic checkpoint publication failed ({outcome['error']})")
        finally:
            if trainer.is_global_zero:
                temporary.unlink(missing_ok=True)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = str(destination)
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    @staticmethod
    def _link_checkpoint(trainer, filepath, linkpath):
        outcome = None
        if trainer.is_global_zero:
            temporary = Path(linkpath).with_name(
                f".{Path(linkpath).name}.{uuid.uuid4().hex}.partial"
            )
            try:
                if os.path.abspath(filepath) != os.path.abspath(linkpath):
                    os.symlink(os.path.relpath(filepath, temporary.parent), temporary)
                    os.replace(temporary, linkpath)
                    _sync_directory(temporary.parent)
                outcome = {"ok": True}
            except Exception as error:
                outcome = {"error": type(error).__name__}
            finally:
                temporary.unlink(missing_ok=True)
        outcome = trainer.strategy.broadcast(outcome)
        if "error" in outcome:
            raise RuntimeError(f"Atomic last-checkpoint publication failed ({outcome['error']})")

    def _remove_checkpoint(self, trainer, filepath):
        # Lightning normally removes the preceding step before updating last.ckpt.
        # Defer deletion so a crash between those operations leaves a usable alias.
        self._pending_removals.append(filepath)

    def _save_none_monitor_checkpoint(self, trainer, monitor_candidates):
        # Include the stable alias in the first serialized callback state. Upstream
        # assigns it only after top-k serialization; resume can otherwise create
        # last-v1.ckpt and later prune the target still referenced by last.ckpt.
        self.last_model_path = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )
        super()._save_none_monitor_checkpoint(trainer, monitor_candidates)

    def _save_last_checkpoint(self, trainer, monitor_candidates):
        # Also normalize older checkpoints that recorded an empty/versioned alias.
        self.last_model_path = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )
        super()._save_last_checkpoint(trainer, monitor_candidates)
        for filepath in self._pending_removals:
            super()._remove_checkpoint(trainer, filepath)
        self._pending_removals.clear()

    def on_train_end(self, trainer, pl_module):
        # max_steps, Timer and an ordinary should_stop all finish here. A crash or
        # SIGKILL cannot promise a final save; the previous periodic file remains.
        if trainer.global_step > 0 and self._last_global_step_saved != trainer.global_step:
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)
        super().on_train_end(trainer, pl_module)

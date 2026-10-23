"""Step-based latest checkpoints with an explicit final save.

Lightning owns checkpoint serialization, optimizer/loop state and retention.
This adapter only makes its unmonitored step policy also run at normal train end.
It does not upload files or checkpoint iterable-loader positions.
"""

from lightning.pytorch.callbacks import ModelCheckpoint


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

    def on_train_end(self, trainer, pl_module):
        # max_steps, Timer and an ordinary should_stop all finish here. A crash or
        # SIGKILL cannot promise a final save; the previous periodic file remains.
        if trainer.global_step > 0 and self._last_global_step_saved != trainer.global_step:
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)
        super().on_train_end(trainer, pl_module)

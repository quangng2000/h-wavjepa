"""Early durable saves and safe, finite stdout progress on production runs."""

import json
import time
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.segment_checkpoint import DurableStepCheckpoint
from scripts.training_progress import TrainingProgress


def trainer_state(step=0, **changes):
    return SimpleNamespace(
        **({"global_step": step, "is_global_zero": True, "callback_metrics": {}} | changes)
    )


def test_progress_counts_optimizer_updates_from_resume_and_reports_wall_time(monkeypatch, capsys):
    clock = iter([10.0, 60.0, 160.0])
    monkeypatch.setattr("scripts.training_progress.time.monotonic", lambda: next(clock))
    callback = TrainingProgress(every_n_steps=100)
    trainer = trainer_state(
        250, callback_metrics={"train/loss": torch.tensor(1.25, requires_grad=True)}
    )
    callback.on_train_start(trainer, None)
    trainer.global_step = 349
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert not capsys.readouterr().out
    trainer.global_step = 350
    callback.on_train_batch_end(trainer, None, None, None, 1)
    first = json.loads(capsys.readouterr().out)
    assert first == {
        "event": "training_progress",
        "global_step": 350,
        "elapsed_seconds": 50.0,
        "interval_seconds": 50.0,
        "interval_updates": 100,
        "updates_per_second": 2.0,
        "loss": 1.25,
    }
    # Repeated callbacks at the same optimizer step (e.g. accumulation) do not log.
    callback.on_train_batch_end(trainer, None, None, None, 2)
    assert not capsys.readouterr().out
    trainer.global_step = 450
    callback.on_train_batch_end(trainer, None, None, None, 3)
    second = json.loads(capsys.readouterr().out)
    assert second["elapsed_seconds"] == 150.0
    assert second["updates_per_second"] == 1.0
    assert not any(isinstance(item, torch.Tensor) for item in vars(callback).values())


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), torch.tensor([1.0, 2.0]), None, True])
def test_non_scalar_or_nonfinite_loss_is_json_null(loss, monkeypatch, capsys):
    clock = iter([0.0, 1.0])
    monkeypatch.setattr("scripts.training_progress.time.monotonic", lambda: next(clock))
    callback = TrainingProgress(every_n_steps=1)
    trainer = trainer_state(callback_metrics={"train/loss": loss})
    callback.on_train_start(trainer, None)
    trainer.global_step = 1
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert json.loads(capsys.readouterr().out)["loss"] is None


def test_nonzero_ranks_do_not_report_progress(capsys):
    callback = TrainingProgress(every_n_steps=1)
    trainer = trainer_state(is_global_zero=False)
    callback.on_train_start(trainer, None)
    trainer.global_step = 100
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert not capsys.readouterr().out


def test_opt_in_reports_nonzero_rank_optimizer_progress(monkeypatch, capsys):
    clock = iter([0.0, 2.0])
    monkeypatch.setattr("scripts.training_progress.time.monotonic", lambda: next(clock))
    callback = TrainingProgress(every_n_steps=100, all_ranks=True)
    trainer = trainer_state(is_global_zero=False, global_rank=3, world_size=4)
    callback.on_train_start(trainer, None)
    trainer.global_step = 100
    callback.on_train_batch_end(trainer, None, None, None, 0)
    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "training_rank_progress"
    assert record["rank"] == 3
    assert record["world_size"] == 4
    assert record["global_step"] == 100


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_progress_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="positive integer"):
        TrainingProgress(interval)


class TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return (self.weight * batch).square().mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def checkpoint_callback(tmp_path, **kwargs):
    return DurableStepCheckpoint(
        repository_options={},
        segment_id="segment-001",
        data_seed=42,
        receipt_path=str(tmp_path / "receipt.json"),
        transfer_timeout_seconds=30,
        hard_deadline=time.time() + 600,
        dirpath=tmp_path / "checkpoints",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("early", "cadence", "expected"),
    [(2, 4, [2, 4]), (2, 2, [2, 4]), (None, 4, [4])],
)
def test_early_save_preserves_full_state_and_does_not_duplicate_cadence(
    early, cadence, expected, tmp_path, monkeypatch, capsys
):
    callback = checkpoint_callback(
        tmp_path, first_checkpoint_step=early, every_n_train_steps=cadence
    )
    published = []

    def publish(path, step):
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["global_step"] == step
        assert int(state["optimizer_states"][0]["state"][0]["step"]) == step
        assert state["loops"]
        published.append(step)
        return {
            "checkpoint": {"step": step, "sha256": "a" * 64, "bytes": path.stat().st_size},
            "private_unused_metadata": "not-for-stdout",
        }

    monkeypatch.setattr(callback, "_publish", publish)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=4,
        callbacks=[callback],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(TinyModel(), train_dataloaders=DataLoader(torch.ones(8, 1), batch_size=1))
    assert published == expected
    reports = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert [record["step"] for record in reports] == expected
    assert all(set(record) == {"event", "step", "sha256", "bytes"} for record in reports)
    assert all(record["event"] == "checkpoint_published" for record in reports)
    assert all(record["bytes"] > 0 for record in reports)


def test_failed_publication_never_logs_success(tmp_path, monkeypatch, capsys):
    callback = checkpoint_callback(tmp_path, first_checkpoint_step=1, every_n_train_steps=5000)

    def fail(path, step):
        raise OSError("private SDK failure detail")

    monkeypatch.setattr(callback, "_publish", fail)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        callbacks=[callback],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Durable checkpoint publication failed \\(OSError\\)"):
        trainer.fit(TinyModel(), train_dataloaders=DataLoader(torch.ones(8, 1), batch_size=1))
    output = capsys.readouterr().out
    assert "checkpoint_published" not in output
    assert "private SDK failure detail" not in output
    assert callback.last_receipt is None


@pytest.mark.parametrize("step", [0, -1, True, 1.5])
def test_early_checkpoint_requires_positive_integer(step, tmp_path):
    with pytest.raises(ValueError, match="first_checkpoint_step"):
        checkpoint_callback(tmp_path, first_checkpoint_step=step)

"""Real Lightning lifecycle checks on an infinite training iterator."""

from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from torch.utils.data import DataLoader, IterableDataset

from src.callbacks.step_checkpoint import StepCheckpoint
from src.utils.instantiators import instantiate_callbacks


class InfiniteOnes(IterableDataset):
    def __iter__(self):
        while True:
            yield torch.ones(1)


class TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def training_step(self, batch, batch_idx):
        return (batch * self.weight).square().mean()

    def validation_step(self, batch, batch_idx):
        # Worsens every validation: only the first validation can be "best".
        self.log("val/loss", float(self.global_step), on_epoch=True, batch_size=len(batch))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


class ObservedStepCheckpoint(StepCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_steps = []

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        self.saved_steps.append(trainer.global_step)


def trainer_options(tmp_path):
    return dict(
        accelerator="cpu",
        devices=1,
        min_epochs=0,
        max_epochs=-1,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=2,
        default_root_dir=tmp_path,
    )


@pytest.mark.integration
def test_latest_and_final_persist_when_validation_never_improves(tmp_path):
    latest = ObservedStepCheckpoint(
        dirpath=tmp_path / "latest",
        filename="step_{step:07d}",
        every_n_train_steps=2,
        auto_insert_metric_name=False,
    )
    best = ModelCheckpoint(
        dirpath=tmp_path / "best",
        filename="best_step_{step:07d}",
        monitor="val/loss",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(max_steps=5, callbacks=[latest, best], **trainer_options(tmp_path))
    model = TinyModel()
    trainer.fit(
        model,
        train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2),
        val_dataloaders=DataLoader([torch.ones(1)] * 2, batch_size=2),
    )
    assert latest.saved_steps == [2, 4, 5]
    assert trainer.global_step == 5
    assert Path(latest.last_model_path).is_symlink()
    latest_state = torch.load(latest.last_model_path, map_location="cpu", weights_only=True)
    best_state = torch.load(best.best_model_path, map_location="cpu", weights_only=True)
    assert latest_state["global_step"] == 5
    assert best_state["global_step"] == 2
    assert latest_state["optimizer_states"][0]["state"]
    torch.testing.assert_close(latest_state["state_dict"]["weight"], model.weight.detach())
    # One regular latest file plus one best file; last is an alias, not another copy.
    files = [path for path in tmp_path.rglob("*.ckpt") if not path.is_symlink()]
    assert len(files) == 2


@pytest.mark.integration
def test_official_timer_stops_infinite_input_and_saves_final_state(tmp_path, monkeypatch):
    latest = StepCheckpoint(dirpath=tmp_path / "latest", every_n_train_steps=10)
    timer = Timer(duration={"seconds": 1}, interval="step", verbose=False)
    trainer = L.Trainer(
        max_steps=20, callbacks=[timer, latest], limit_val_batches=0, **trainer_options(tmp_path)
    )
    # Advance Timer's observed clock after step 3, without wall-clock sleeps or
    # replacing the official stop callback. min_epochs=0 lets should_stop finish.
    monkeypatch.setattr(
        timer, "time_elapsed", lambda stage="train": 2.0 if trainer.global_step >= 3 else 0.0
    )
    trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert trainer.global_step == 3
    state = torch.load(latest.last_model_path, map_location="cpu", weights_only=True)
    assert state["global_step"] == 3
    assert state["optimizer_states"][0]["state"]


@pytest.mark.parametrize("experiment", ["audioset_h100_fast", "audioset_h100_four"])
def test_fast_preset_callbacks_construct_with_progress_disabled(make_config, experiment):
    cfg = make_config([f"experiment={experiment}"])
    callbacks = instantiate_callbacks(cfg.callbacks)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=False,
    )
    assert trainer.progress_bar_callback is None


def test_step_checkpoint_preset_composes_latest_and_optional_best(make_config):
    cfg = make_config(
        ["experiment=audioset_h100_fast", "callbacks=step_checkpoint", "trainer.min_epochs=0"]
    )
    callbacks = instantiate_callbacks(cfg.callbacks)
    assert isinstance(callbacks[0], StepCheckpoint)
    assert callbacks[0].monitor is None
    assert callbacks[0].save_top_k == 1
    assert callbacks[0].save_last == "link"
    assert callbacks[0]._every_n_train_steps == 5000
    assert cfg.trainer.min_epochs == 0
    assert cfg.callbacks.best_checkpoint.monitor == "val/loss"
    assert cfg.callbacks.rich_progress_bar is None


@pytest.mark.integration
def test_fast_optimizer_keeps_lightning_bf16_gradient_clipping(make_config, tmp_path):
    cfg = make_config(["experiment=audioset_h100_fast"])
    model = TinyModel()
    model.configure_optimizers = lambda: torch.optim.AdamW(
        model.parameters(), fused=cfg.model.optimizer.fused
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=1,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        default_root_dir=tmp_path,
    )
    before = model.weight.detach().clone()
    trainer.fit(model, train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert trainer.gradient_clip_val == 1.0
    assert trainer.global_step == 1
    assert not torch.equal(model.weight.detach(), before)


@pytest.mark.integration
def test_interrupted_checkpoint_write_preserves_previous_last(tmp_path, monkeypatch):
    latest = StepCheckpoint(dirpath=tmp_path / "latest", every_n_train_steps=2)
    trainer = L.Trainer(
        max_steps=5, callbacks=[latest], limit_val_batches=0, **trainer_options(tmp_path)
    )
    original = trainer.save_checkpoint

    def interrupted(filepath, *args, **kwargs):
        if trainer.global_step == 4:
            Path(filepath).write_bytes(b"incomplete serialization")
            raise OSError("simulated interrupted write")
        original(filepath, *args, **kwargs)

    monkeypatch.setattr(trainer, "save_checkpoint", interrupted)
    with pytest.raises(OSError, match="interrupted write"):
        trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    state = torch.load(latest.last_model_path, weights_only=True)
    assert state["global_step"] == 2
    assert Path(latest.last_model_path).is_symlink()
    assert not list((tmp_path / "latest").glob("*.partial"))
    assert len([p for p in (tmp_path / "latest").glob("*.ckpt") if not p.is_symlink()]) == 1


@pytest.mark.integration
def test_failed_alias_replacement_keeps_previous_target_until_publication(tmp_path, monkeypatch):
    import os

    latest = StepCheckpoint(dirpath=tmp_path / "latest", every_n_train_steps=2)
    trainer = L.Trainer(
        max_steps=5, callbacks=[latest], limit_val_batches=0, **trainer_options(tmp_path)
    )
    original = os.replace

    def interrupted(source, destination):
        if trainer.global_step == 4 and Path(destination).name == "last.ckpt":
            raise OSError("simulated alias replacement failure")
        return original(source, destination)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(RuntimeError, match="last-checkpoint publication"):
        trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert torch.load(latest.last_model_path, weights_only=True)["global_step"] == 2
    # The completed replacement remains available too; retention waits for alias publication.
    paths = [p for p in (tmp_path / "latest").glob("*.ckpt") if not p.is_symlink()]
    assert sorted(torch.load(p, weights_only=True)["global_step"] for p in paths) == [2, 4]
    assert not list((tmp_path / "latest").glob("*.partial"))


@pytest.mark.integration
def test_first_checkpoint_resumes_in_same_directory_without_dangling_last_alias(tmp_path):
    directory = tmp_path / "latest"
    first = StepCheckpoint(dirpath=directory, every_n_train_steps=2)
    trainer = L.Trainer(
        max_steps=2, callbacks=[first], limit_val_batches=0, **trainer_options(tmp_path)
    )
    trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    saved = torch.load(directory / "last.ckpt", weights_only=True)
    assert saved["callbacks"][first.state_key]["last_model_path"] == str(directory / "last.ckpt")
    resumed = StepCheckpoint(dirpath=directory, every_n_train_steps=2)
    trainer = L.Trainer(
        max_steps=4, callbacks=[resumed], limit_val_batches=0, **trainer_options(tmp_path)
    )
    trainer.fit(
        TinyModel(),
        train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2),
        ckpt_path=str(directory / "last.ckpt"),
        weights_only=True,
    )
    assert Path(resumed.last_model_path) == directory / "last.ckpt"
    assert (directory / "last.ckpt").is_symlink()
    assert torch.load(directory / "last.ckpt", weights_only=True)["global_step"] == 4
    assert not list(directory.glob("last-v*.ckpt"))
    assert len([p for p in directory.glob("*.ckpt") if not p.is_symlink()]) == 1

"""Official schedule reuse and independent PyTorch predictor initialization."""

from copy import deepcopy
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from test_hierarchical_dynamics import checkpoint_for, model_factory, waveform
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.models.components.temporal_predictor import TemporalPredictor
from src.vendor.wavjepa.jepa import JEPA


def test_temporal_predictor_layers_have_independent_initial_values_and_gradients():
    predictor = TemporalPredictor(16, 5, layers=3, heads=2)
    layers = predictor.transformer.layers
    for left, right in zip(layers[:-1], layers[1:], strict=True):
        assert not torch.equal(left.self_attn.in_proj_weight, right.self_attn.in_proj_weight)
        assert not torch.equal(left.linear1.weight, right.linear1.weight)
    predictor(torch.randn(2, 5, 16), torch.arange(5)).square().mean().backward()
    assert all(layer.self_attn.in_proj_weight.grad.abs().sum() > 0 for layer in layers)


@pytest.mark.parametrize(
    "invalid",
    [
        {"lr_schedule": "other"},
        {"lr_warmup_steps": -1},
        {"lr_warmup_steps": True},
        {"lr_total_steps": 0},
        {"lr_total_steps": 1.5},
        {"ema_end_decay": 0.8},
        {"ema_end_decay": 1.0},
        {"ema_end_decay": float("nan")},
        {"ema_anneal_end_step": 0},
        {"ema_anneal_end_step": 2},
    ],
)
def test_invalid_schedule_configuration_is_rejected(invalid):
    with pytest.raises(ValueError):
        model_factory(**invalid)


def test_warmup_cosine_matches_the_official_transformers_scheduler():
    model = model_factory(lr_schedule="cosine", lr_warmup_steps=2, lr_total_steps=6)
    configuration = model.configure_optimizers()
    optimizer = configuration["optimizer"]
    scheduler = configuration["lr_scheduler"]["scheduler"]
    reference_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=0.001)
    reference = get_cosine_schedule_with_warmup(reference_optimizer, 2, 6)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        assert scheduler.get_last_lr() == reference.get_last_lr()
        optimizer.step()
        reference_optimizer.step()
        scheduler.step()
        reference.step()
    assert rates[:3] == pytest.approx([0.0, 0.0005, 0.001])
    assert rates[3] > rates[4] > rates[5] > scheduler.get_last_lr()[0]
    assert model.ema_updates.item() == 6


def test_ema_annealing_matches_official_linear_schedule_and_stops_at_end():
    model = model_factory(ema_end_decay=0.99, ema_anneal_end_step=4)
    observed = []
    for step in range(7):
        observed.append(model.current_ema_decay())
        model.update_teacher()
        reference = SimpleNamespace(
            global_step=step,
            ema_end_step=4,
            hparams=SimpleNamespace(ema_decay=0.9, ema_end_decay=0.99),
        )
        expected = JEPA._get_ema_decay(reference)
        assert observed[-1] == pytest.approx(expected)
    assert observed == pytest.approx([0.9, 0.9225, 0.945, 0.9675, 0.99, 0.99, 0.99])


def test_old_schema_one_checkpoints_keep_constant_schedules():
    model = model_factory()
    checkpoint = checkpoint_for(model)
    for key in model.OPTIONAL_CONFIG_DEFAULTS:
        checkpoint[model.CHECKPOINT_KEY]["dynamics"].pop(key)
    model.on_load_checkpoint(checkpoint)
    assert model.current_ema_decay() == 0.9
    assert isinstance(model.configure_optimizers(), torch.optim.Optimizer)


@pytest.mark.parametrize("bad", [{}, {"total_steps": 0, "warmup_steps": 0, "ema_end_step": 1}])
def test_corrupt_resolved_training_schedule_is_rejected(bad):
    model = model_factory()
    checkpoint = checkpoint_for(model)
    checkpoint[model.SCHEDULE_CHECKPOINT_KEY] = bad
    with pytest.raises(ValueError, match="invalid checkpoint training schedule"):
        model.on_load_checkpoint(checkpoint)


def trainer_options(tmp_path):
    return dict(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        accumulate_grad_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )


class StopAfterSteps(L.Callback):
    def __init__(self, steps):
        self.steps = steps

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.global_step >= self.steps:
            trainer.should_stop = True


@pytest.mark.integration
@pytest.mark.parametrize("total_steps", [None, 4])
@pytest.mark.parametrize("resume_stop", [3, 4])
def test_lightning_resume_preserves_schedule_ema_and_optimizer_with_accumulation(
    tmp_path, total_steps, resume_stop
):
    torch.manual_seed(29)
    model = model_factory(
        lr_schedule="cosine", lr_warmup_steps=1, lr_total_steps=total_steps, ema_end_decay=0.99
    )
    loader = DataLoader([{"waveform": item} for item in waveform(8)], batch_size=2)
    initial = deepcopy(model.state_dict())
    trainer = L.Trainer(max_steps=4, callbacks=[StopAfterSteps(2)], **trainer_options(tmp_path))
    trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == model.ema_updates.item() == 2
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.00075)
    assert model.current_ema_decay() == pytest.approx(0.945)
    checkpoint_path = tmp_path / "scheduled.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint[model.SCHEDULE_CHECKPOINT_KEY] == {
        "total_steps": 4,
        "warmup_steps": 1,
        "ema_end_step": 4,
    }

    restored = model_factory(
        lr_schedule="cosine", lr_warmup_steps=1, lr_total_steps=total_steps, ema_end_decay=0.99
    )
    resumed = L.Trainer(max_steps=resume_stop, **trainer_options(tmp_path))
    resumed.fit(restored, train_dataloaders=loader, ckpt_path=str(checkpoint_path))
    assert resumed.global_step == restored.ema_updates.item() == resume_stop
    assert restored.current_ema_decay() == pytest.approx(0.9675 if resume_stop == 3 else 0.99)
    assert resumed.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        0.00025 if resume_stop == 3 else 0
    )
    assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == resume_stop
    assert not torch.equal(initial["abstractions.0.0.weight"], restored.abstractions[0][0].weight)


def test_checkpoint_budget_is_kept_when_resuming_with_a_different_stopping_point():
    model = model_factory(lr_schedule="cosine", lr_warmup_steps=None, ema_end_decay=0.99)
    model._trainer = SimpleNamespace(estimated_stepping_batches=20)
    model.configure_optimizers()
    checkpoint = checkpoint_for(model)
    restored = model_factory(lr_schedule="cosine", lr_warmup_steps=None, ema_end_decay=0.99)
    restored.on_load_checkpoint(checkpoint)
    restored._trainer = SimpleNamespace(estimated_stepping_batches=10)
    restored.configure_optimizers()
    restored.on_train_start()
    assert restored._resolved_training_schedule == {
        "total_steps": 20,
        "warmup_steps": 2,
        "ema_end_step": 20,
    }
    restored._trainer.estimated_stepping_batches = 21
    with pytest.raises(ValueError, match="exceeds the saved cosine schedule budget"):
        restored.on_train_start()


@pytest.mark.integration
def test_automatic_schedule_uses_optimizer_steps_after_gradient_accumulation(tmp_path):
    model = model_factory(lr_schedule="cosine", lr_warmup_steps=None, ema_end_decay=0.99)
    loader = DataLoader([{"waveform": item} for item in waveform(8)], batch_size=2)
    trainer = L.Trainer(max_epochs=1, **trainer_options(tmp_path))
    trainer.fit(model, train_dataloaders=loader)
    assert model._resolved_training_schedule == {
        "total_steps": 2,
        "warmup_steps": 0,
        "ema_end_step": 2,
    }
    assert trainer.global_step == model.ema_updates.item() == 2
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0)


@pytest.mark.integration
def test_nonfinite_gradients_advance_neither_schedule_nor_teacher(tmp_path):
    model = model_factory(lr_schedule="cosine", lr_total_steps=2, ema_end_decay=0.99)
    handle = next(model.abstractions.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    trainer = L.Trainer(max_steps=2, **trainer_options(tmp_path))
    loader = DataLoader([{"waveform": item} for item in waveform(4)], batch_size=2)
    try:
        with pytest.raises(FloatingPointError, match="non-finite gradient"):
            trainer.fit(model, train_dataloaders=loader)
    finally:
        handle.remove()
    assert model.ema_updates.item() == trainer.global_step == 0
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 0
    assert trainer.optimizers[0].state == {}

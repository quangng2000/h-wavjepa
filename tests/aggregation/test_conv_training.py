"""Short CPU training verifies optimizer and EMA resume for new Conv weights."""

import lightning as L
import pytest
import torch
from test_hierarchical_dynamics import model_factory, waveform
from test_training_schedule import trainer_options
from torch.utils.data import DataLoader


def make_model():
    return model_factory(
        aggregation_mode="conv1d",
        aggregation_windows=[1, 2, 2],
        lr_schedule="cosine",
        lr_total_steps=4,
        lr_warmup_steps=1,
        ema_end_decay=0.99,
        ema_anneal_end_step=4,
    )


def fit_steps(model, loader, directory, steps, checkpoint=None):
    trainer = L.Trainer(max_steps=steps, **trainer_options(directory))
    trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint)
    assert trainer.global_step == model.ema_updates.item() == steps
    for abstraction in model.abstractions:
        parameter = next(abstraction[0].parameters())
        state = trainer.optimizers[0].state[parameter]
        assert state["step"] == steps
        assert torch.isfinite(state["exp_avg"]).all()
        assert state["exp_avg"].abs().sum() > 0
    return trainer


@pytest.mark.integration
def test_convolution_optimizer_and_ema_continue_after_checkpoint(tmp_path):
    torch.manual_seed(29)
    loader = DataLoader([{"waveform": item} for item in waveform(8)], batch_size=2)
    model = make_model()
    initial = next(model.abstractions[0][0].parameters()).detach().clone()
    trainer = fit_steps(model, loader, tmp_path, 2)
    checkpoint = tmp_path / "resume-conv.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = make_model()
    resumed = fit_steps(restored, loader, tmp_path, 3, str(checkpoint))
    assert not torch.equal(next(restored.abstractions[0][0].parameters()), initial)
    assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 3
    assert restored.current_ema_decay() == pytest.approx(0.9675)

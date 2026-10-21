"""Versioned model contracts and deterministic, fail-fast research evaluation."""

import json
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.components.hierarchy import HierarchyEncoder, patchify
from src.models.hwavjepa_module import HWavJEPAModule


def model_factory(**overrides) -> HWavJEPAModule:
    options = {
        "encoder": HierarchyEncoder(patch_size=8, widths=(8, 12, 16), pool_factors=(2, 2)),
        "optimizer": partial(torch.optim.AdamW, lr=0.001),
        "protected_dim": 8,
        "input_sample_rate": 16_000,
    }
    return HWavJEPAModule(**(options | overrides))


def model_checkpoint(model: HWavJEPAModule) -> dict:
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    return checkpoint


def test_checkpoint_metadata_is_plain_and_reconstructs_encoder_and_model(tmp_path):
    source = model_factory().eval()
    checkpoint = model_checkpoint(source)
    metadata = source.validated_checkpoint_metadata(checkpoint)
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["encoder"]["pool_factors"] == [2, 2]
    assert metadata["module"]["input_sample_rate"] == 16_000
    path = tmp_path / "weights.ckpt"
    torch.save(checkpoint, path)
    restored_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = HWavJEPAModule(
        encoder=HierarchyEncoder(**metadata["encoder"]),
        optimizer=partial(torch.optim.AdamW),
        **metadata["module"],
    ).eval()
    restored.on_load_checkpoint(restored_checkpoint)
    restored.load_state_dict(restored_checkpoint["state_dict"], strict=True)
    waveform = torch.randn(2, 128)
    for actual, expected in zip(
        restored.encode(waveform)["levels"], source.encode(waveform)["levels"], strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"mask_ratio": 0.75}, "module.mask_ratio"),
        ({"mask_group_size": 8}, "module.mask_group_size"),
        ({"ema_decay": 0.5}, "module.ema_decay"),
        ({"protection_weight": 2.0}, "module.protection_weight"),
        ({"protected_dim": 12}, "module.protected_dim"),
        ({"prediction_weights": [1.0, 0.5, 1.0]}, "module.prediction_weights"),
        ({"eval_mask_seed": 17}, "module.eval_mask_seed"),
        ({"input_sample_rate": 48_000}, "module.input_sample_rate"),
    ],
)
def test_checkpoint_rejects_changed_model_semantics(changes, field):
    checkpoint = model_checkpoint(model_factory())
    with pytest.raises(ValueError, match=field):
        model_factory(**changes).on_load_checkpoint(checkpoint)


def test_checkpoint_rejects_same_shaped_pooling_change():
    original = model_factory()
    changed = model_factory(
        encoder=HierarchyEncoder(patch_size=8, widths=(8, 12, 16), pool_factors=(1, 4))
    )
    # A tensor-only load accepts this mathematically different encoder.
    changed.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(ValueError, match="encoder.pool_factors"):
        changed.on_load_checkpoint(model_checkpoint(original))


@pytest.mark.parametrize("kind", ["missing", "future", "unknown_field", "semantics", "nan"])
def test_checkpoint_rejects_missing_or_invalid_metadata(kind):
    model = model_factory()
    checkpoint = model_checkpoint(model)
    if kind == "missing":
        checkpoint.pop("hwavjepa")
    elif kind == "future":
        checkpoint["hwavjepa"]["schema_version"] = 999
    elif kind == "unknown_field":
        checkpoint["hwavjepa"]["encoder"]["unrecognized"] = 1
    elif kind == "semantics":
        checkpoint["hwavjepa"]["semantics"]["objective"] = "different_loss"
    else:
        checkpoint["hwavjepa"]["module"]["mask_ratio"] = float("nan")
    with pytest.raises(ValueError):
        model.on_load_checkpoint(checkpoint)


def test_metadata_validation_returns_independent_copy():
    model = model_factory()
    checkpoint = model_checkpoint(model)
    saved = deepcopy(checkpoint["hwavjepa"])
    result = model.validated_checkpoint_metadata(checkpoint)
    result["encoder"]["widths"][0] = 123
    assert checkpoint["hwavjepa"] == saved


def test_eval_loss_and_masks_repeat_without_consuming_global_rng():
    model = model_factory().eval()
    waveform = torch.randn(3, 256)
    before = torch.get_rng_state().clone()
    first = model.compute_loss(waveform)
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(100)
    before_second = torch.get_rng_state().clone()
    second = model.compute_loss(waveform)
    assert torch.equal(torch.get_rng_state(), before_second)
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    assert torch.equal(first["patch_mask"], second["patch_mask"])


@pytest.mark.parametrize("stage", ["validation_step", "test_step"])
def test_lightning_evaluation_uses_repeatable_batch_index_seeds(stage, monkeypatch):
    model = model_factory().eval()
    waveform = torch.randn(3, 256)
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    before = torch.get_rng_state().clone()
    first = getattr(model, stage)({"waveform": waveform}, 7)
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(100)
    second = getattr(model, stage)({"waveform": waveform}, 7)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    expected = model.compute_loss(
        waveform, model.make_patch_mask(waveform, seed=model.eval_mask_seed + 7)
    )["loss"]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    assert not torch.equal(
        model.make_patch_mask(waveform, seed=7), model.make_patch_mask(waveform, seed=8)
    )


def test_training_masks_remain_stochastic():
    model = model_factory().train()
    waveform = torch.randn(3, 256)
    before = torch.get_rng_state().clone()
    first = model.make_patch_mask(waveform)
    assert not torch.equal(torch.get_rng_state(), before)
    assert not torch.equal(first, model.make_patch_mask(waveform))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_waveform_is_rejected_before_model_computation(invalid):
    model = model_factory()
    waveform = torch.randn(2, 128)
    waveform[0, 0] = invalid
    with pytest.raises(ValueError, match="samples must be finite"):
        patchify(waveform, 8)
    with pytest.raises(ValueError, match="samples must be finite"):
        model.compute_loss(waveform)
    with pytest.raises(ValueError, match="samples must be finite"):
        model.encode(waveform)


def test_nonfinite_loss_is_rejected_before_backward():
    model = model_factory()
    with torch.no_grad():
        model.protected_decoder.bias.fill_(1e30)
    with pytest.raises(FloatingPointError, match="non-finite H-WavJEPA loss"):
        model.compute_loss(torch.randn(2, 128))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.ema_updates.item() == 0


@pytest.mark.parametrize("seed", [-1, 2**63, 0.5, True])
def test_evaluation_seed_validation(seed):
    with pytest.raises(ValueError, match="eval_mask_seed"):
        model_factory(eval_mask_seed=seed)


@pytest.mark.parametrize("sample_rate", [-1, 0, 16_000.5, True])
def test_sample_rate_contract_validation(sample_rate):
    with pytest.raises(ValueError, match="input_sample_rate"):
        model_factory(input_sample_rate=sample_rate)


@pytest.mark.parametrize(
    "options", [{"patch_size": 8.5}, {"widths": [8.5]}, {"pool_factors": [2.5, 2]}]
)
def test_encoder_configuration_never_silently_truncates_fractional_dimensions(options):
    with pytest.raises(ValueError, match="positive integer"):
        HierarchyEncoder(**options)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_gradients_fail_with_parameter_name_and_preserve_state(invalid):
    model = model_factory()
    optimizer = model.configure_optimizers()
    parameter = model.encoder.patch_encoder[0].weight
    parameter.grad = torch.full_like(parameter, invalid)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(FloatingPointError, match="encoder.patch_encoder.0.weight"):
        model.on_before_optimizer_step(optimizer)
        optimizer.step()
    assert optimizer.state == {}
    assert model.ema_updates.item() == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_finite_gradient_guard_allows_optimizer_and_ema_step():
    model = model_factory()
    optimizer = model.configure_optimizers()
    model.compute_loss(torch.randn(2, 128))["loss"].backward()
    model.on_before_optimizer_step(optimizer)
    optimizer.step()
    assert optimizer.state
    assert model.ema_updates.item() == 1


def test_gradient_guard_defers_to_precision_plugin_grad_scaler(monkeypatch):
    model = model_factory()
    optimizer = model.configure_optimizers()
    parameter = model.encoder.patch_encoder[0].weight
    parameter.grad = torch.full_like(parameter, float("inf"))
    precision = SimpleNamespace(scaler=object())
    monkeypatch.setattr(model, "_trainer", SimpleNamespace(precision_plugin=precision))
    # The GradScaler owns whether to step. This hook must not raise on AMP's
    # recoverable overflow; an actual mixed-precision accelerator is not claimed.
    model.on_before_optimizer_step(optimizer)
    assert model.ema_updates.item() == 0
    precision.scaler = None
    with pytest.raises(FloatingPointError, match="encoder.patch_encoder.0.weight"):
        model.on_before_optimizer_step(optimizer)


@pytest.mark.integration
def test_lightning_rejects_bad_backward_before_adamw_or_ema_mutates(tmp_path):
    model = model_factory()
    parameter = model.encoder.patch_encoder[0].weight
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader([{"waveform": torch.randn(128)}], batch_size=1)
    try:
        with pytest.raises(FloatingPointError, match="encoder.patch_encoder.0.weight"):
            trainer.fit(model, train_dataloaders=loader)
    finally:
        handle.remove()
    assert trainer.optimizers[0].state == {}
    assert model.ema_updates.item() == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class MockFusedOptimizer(torch.optim.Optimizer):
    """CPU-only simulation of the fused step's found_inf/skip contract."""

    def __init__(self, params):
        super().__init__(params, {"lr": 0.1})
        self.found_inf = torch.tensor(1.0)

    @torch.no_grad()
    def step(self, closure=None):
        if self.found_inf.ne(0).any():
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-group["lr"])


def test_mock_fused_optimizer_hook_skips_ema_on_found_inf_and_updates_on_zero():
    model = model_factory(optimizer=MockFusedOptimizer)
    optimizer = model.configure_optimizers()
    parameter = model.encoder.patch_encoder[0].weight
    parameter.grad = torch.ones_like(parameter)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer.step()  # Post-hook runs even though the mock kernel skips.
    assert model.ema_updates.item() == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    optimizer.found_inf.zero_()
    optimizer.step()
    assert model.ema_updates.item() == 1
    assert not torch.equal(parameter, before["encoder.patch_encoder.0.weight"])


def test_cpu_fused_adamw_grad_scaler_skip_does_not_advance_ema():
    model = model_factory(optimizer=partial(torch.optim.AdamW, lr=0.01, fused=True))
    try:
        optimizer = model.configure_optimizers()
    except (RuntimeError, NotImplementedError) as error:
        if "fused" not in str(error).lower():
            raise
        pytest.skip(f"This PyTorch build does not support CPU fused AdamW: {error}")
    scaler = torch.amp.GradScaler("cpu")
    parameter = model.encoder.patch_encoder[0].weight
    scaler.scale(parameter.sum()).backward()
    parameter.grad.fill_(float("inf"))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    try:
        scaler.step(optimizer)
    except (RuntimeError, NotImplementedError) as error:
        message = str(error).lower()
        if "fused" not in message or not any(
            phrase in message for phrase in ("not support", "not implement", "could not run")
        ):
            raise
        pytest.skip(f"This PyTorch build does not support CPU fused AdamW: {error}")
    scaler.update()
    assert model.ema_updates.item() == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().mean()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert model.ema_updates.item() == 1
    assert not torch.equal(parameter, before["encoder.patch_encoder.0.weight"])

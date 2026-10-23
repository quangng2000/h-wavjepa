"""Performance controls preserve training state and failure semantics."""

from copy import deepcopy

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from test_hierarchical_dynamics import model_factory, waveform
from torch.utils.data import DataLoader

from src.utils.runtime import require_finite_gradients, validate_config


def test_gradient_guard_uses_one_host_boolean_for_many_parameters(monkeypatch):
    parameters = [torch.nn.Parameter(torch.ones(8)) for _ in range(24)]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    original = torch.Tensor.__bool__
    conversions = []

    def observe(tensor):
        conversions.append(tensor.shape)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", observe)
    require_finite_gradients(optimizer)
    assert len(conversions) == 1


@pytest.mark.parametrize("nonfinite", [float("inf"), -float("inf"), float("nan")])
def test_gradient_guard_catches_nonfinite_values_in_later_groups(nonfinite):
    parameters = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    optimizer = torch.optim.SGD([{"params": [parameter]} for parameter in parameters], lr=0.1)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    parameters[-1].grad[-1] = nonfinite
    with pytest.raises(FloatingPointError, match="optimizer and EMA update aborted"):
        require_finite_gradients(optimizer)


def test_gradient_guard_accepts_large_finite_and_absent_gradients():
    first, unused = [torch.nn.Parameter(torch.ones(8)) for _ in range(2)]
    optimizer = torch.optim.SGD([first, unused], lr=0.1)
    require_finite_gradients(optimizer)
    first.grad = torch.full_like(first, torch.finfo(torch.float32).max)
    assert torch.isinf(first.grad.norm())
    require_finite_gradients(optimizer)


def test_gradient_guard_checks_sparse_values_without_densifying():
    parameter = torch.nn.Parameter(torch.ones(1000))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.sparse_coo_tensor([[999]], [3.0], size=(1000,))
    require_finite_gradients(optimizer)
    parameter.grad._values()[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        require_finite_gradients(optimizer)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_metric_interval_is_validated(interval, make_config):
    with pytest.raises(ValueError, match="training_metric_interval"):
        model_factory(training_metric_interval=interval)
    cfg = make_config(["experiment=audioset_h100_fast"])
    cfg.model.training_metric_interval = interval
    with pytest.raises(ValueError, match="training_metric_interval"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("experiment", "devices"), [("audioset_h100_fast", 2), ("audioset_h100_four", 4)]
)
def test_performance_presets_preserve_research_configuration(make_config, experiment, devices):
    baseline = make_config(["experiment=audioset_h100"])
    candidate = make_config([f"experiment={experiment}"])
    validate_config(candidate)
    assert candidate.trainer.devices == devices
    assert candidate.data.batch_size * devices == 64
    assert candidate.trainer.max_steps == candidate.model.lr_total_steps == 375000
    original_model = OmegaConf.to_container(baseline.model, resolve=True)
    candidate_model = OmegaConf.to_container(candidate.model, resolve=True)
    assert candidate_model.pop("training_metric_interval") == 50
    assert candidate_model["localereg_fixed_shape_kmeans"] is True
    candidate_model["localereg_fixed_shape_kmeans"] = False
    assert candidate_model["optimizer"].pop("fused") is False
    assert candidate_model == original_model
    assert candidate.trainer.strategy == baseline.trainer.strategy
    assert candidate.data.require_full_length
    assert candidate.data.train_random_crop and candidate.data.train_drop_last
    assert candidate.data.persistent_workers and candidate.data.prefetch_factor == 2


def test_sharded_audio_loader_uses_same_full_audio_and_ddp_contract(make_config):
    cfg = make_config(["experiment=audioset_h100_fast"])
    cfg.data._target_ = "src.data.sharded_audio_datamodule.ShardedAudioDataModule"
    validate_config(cfg)
    cfg.trainer.use_distributed_sampler = True
    with pytest.raises(ValueError, match="owns rank sharding"):
        validate_config(cfg)
    cfg.trainer.use_distributed_sampler = False
    cfg.trainer.max_steps = -1
    with pytest.raises(ValueError, match="infinite iterator"):
        validate_config(cfg)
    cfg.trainer.max_steps = 375000
    cfg.data.require_full_length = False
    with pytest.raises(ValueError, match="require_full_length"):
        validate_config(cfg)


def test_old_backbone_checkpoints_default_to_dense_masked_tokens():
    model = model_factory()
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    checkpoint[model.CHECKPOINT_KEY]["backbone"].pop("compact_masked_tokens")
    model.on_load_checkpoint(checkpoint)
    assert model.checkpoint_metadata()["backbone"]["compact_masked_tokens"] is False


def test_fused_mixed_precision_clipping_is_rejected_before_allocation(make_config):
    cfg = make_config(["experiment=audioset_h100_fast"])
    cfg.model.optimizer.fused = True
    with pytest.raises(ValueError, match="fused Adam/AdamW with gradient clipping"):
        validate_config(cfg)
    cfg.model.optimizer.fused = False
    validate_config(cfg)
    assert cfg.trainer.gradient_clip_val == 1.0


@pytest.mark.integration
def test_sampled_training_metrics_preserve_updates_and_checkpoint_resume(tmp_path):
    torch.manual_seed(91)
    initial = deepcopy(model_factory().state_dict())
    audio = waveform(8)
    states = []
    log_counts = []
    for interval in (1, 3):
        model = model_factory(training_metric_interval=interval)
        model.load_state_dict(initial)
        calls = []
        original_log = model.log

        def observe_log(name, *args, **kwargs):
            calls.append(name)
            return original_log(name, *args, **kwargs)

        model.log = observe_log
        trainer = L.Trainer(
            accelerator="cpu",
            max_steps=4,
            max_epochs=-1,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            default_root_dir=tmp_path / str(interval),
        )
        torch.manual_seed(123)
        trainer.fit(
            model,
            train_dataloaders=DataLoader([{"waveform": item} for item in audio], batch_size=2),
        )
        states.append(deepcopy(model.state_dict()))
        log_counts.append(calls.count("train/loss"))
        assert int(model.ema_updates) == 4
        if interval == 3:
            checkpoint = {}
            model.on_save_checkpoint(checkpoint)
            # Changing only the observation cadence is a valid resume.
            restored = model_factory(training_metric_interval=7)
            restored.on_load_checkpoint(checkpoint)
            restored.load_state_dict(model.state_dict())
            assert int(restored.ema_updates) == 4
    assert log_counts == [4, 2]
    assert states[0].keys() == states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

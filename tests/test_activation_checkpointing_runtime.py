"""Activation recomputation is a resumable execution choice, not a new recipe."""

from copy import deepcopy

import lightning as L
import pytest
import torch
from test_hierarchical_dynamics import model_factory
from test_multihost_segments import config_plan
from test_wavjepa_compaction import config as backbone_config
from torch.utils.data import DataLoader

from scripts.train_segment import make_config
from src.models.components.wavjepa_backbone import WavJEPABackbone


def model(checkpointing):
    return model_factory(
        backbone=WavJEPABackbone(
            backbone_config(
                input_channels=2,
                compact_masked_tokens=True,
                use_gradient_checkpointing=checkpointing,
            )
        ),
        aggregation_windows=[1, 2, 2],
        temporal_variance_weight=0.1,
        regularizer_kind="localereg",
        localereg_fixed_shape_kmeans=True,
        readout_mode="normalized_top_k",
        masked_crops_per_recording=8,
        masked_crop_samples=256,
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
    )


def batch():
    return {
        "waveform": torch.randn(2, 2, 9 * 128),
        "masked_crops": torch.randn(2, 8, 2, 256),
    }


def assert_tree_close(actual, expected, **tolerances):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_tree_close(actual[name], expected[name], **tolerances)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_tree_close(left, right, **tolerances)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, **tolerances)
    else:
        assert actual == expected


@pytest.mark.parametrize("seed", [12, 70])
def test_full_binaural_temporal_loss_gradients_and_rng_match_without_recomputation(seed):
    torch.manual_seed(seed)
    checkpointed = model(True).train()
    retained = model(False).train()
    retained.load_state_dict(checkpointed.state_dict(), strict=True)
    inputs = batch()
    initial_rng = torch.get_rng_state()
    records = []
    for instance in (checkpointed, retained):
        torch.set_rng_state(initial_rng)
        audio = inputs["waveform"].detach().clone().requires_grad_()
        crops = inputs["masked_crops"].detach().clone().requires_grad_()
        result = instance.compute_loss(audio, masked_crops=crops)
        result["loss"].backward()
        records.append((result, audio.grad, crops.grad, torch.get_rng_state()))
    for name in (
        "loss",
        "masked_loss",
        "prediction_loss",
        "regularization_loss",
        "temporal_variance_loss",
    ):
        torch.testing.assert_close(records[0][0][name], records[1][0][name], rtol=1e-6, atol=1e-7)
    for name in ("prediction_terms", "regularization_terms", "temporal_variance_terms", "targets"):
        assert_tree_close(records[0][0][name], records[1][0][name], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(records[0][1], records[1][1], rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(records[0][2], records[1][2], rtol=1e-4, atol=2e-6)
    assert torch.equal(records[0][3], records[1][3])
    for (name, expected), (other, actual) in zip(
        checkpointed.named_parameters(), retained.named_parameters(), strict=True
    ):
        assert name == other
        if expected.grad is None or actual.grad is None:
            assert expected.grad is None and actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("runtime_flag", [True, False])
def test_resume_preserves_metadata_and_only_allows_valid_boolean_execution_flag(runtime_flag):
    source = model(not runtime_flag)
    state = {}
    source.on_save_checkpoint(state)
    original = deepcopy(state)
    restored = model(runtime_flag)
    restored.on_load_checkpoint(state)
    assert state == original
    assert restored.checkpoint_metadata()["backbone"]["use_gradient_checkpointing"] is runtime_flag
    for invalid in (None, 0, 1, "false"):
        corrupted = deepcopy(state)
        corrupted[source.CHECKPOINT_KEY]["backbone"]["use_gradient_checkpointing"] = invalid
        with pytest.raises(ValueError, match="configuration mismatch"):
            restored.on_load_checkpoint(corrupted)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("backbone", "compact_masked_tokens", False),
        ("backbone", "input_channels", 1),
        ("backbone", "window_samples", 256),
        ("backbone", "average_top_k_layers", 1),
        ("backbone", "encoder_layers", 4),
        ("dynamics", "masked_weight", 0.0),
        ("dynamics", "aggregation_windows", [1, 1, 1]),
        ("dynamics", "masked_crops_per_recording", 4),
        ("dynamics", "lr_total_steps", 100),
    ],
)
def test_runtime_flag_change_never_waives_semantic_mismatches(section, field, value):
    source, restored = model(True), model(False)
    state = {}
    source.on_save_checkpoint(state)
    state[source.CHECKPOINT_KEY][section][field] = value
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.on_load_checkpoint(state)


@pytest.mark.parametrize("resumed_batch", [2, 4])
def test_full_lightning_resume_restores_optimizer_ema_schedule_and_same_next_update(
    tmp_path, resumed_batch
):
    torch.manual_seed(20)
    inputs = batch()
    loader = DataLoader(
        [{name: values[index % 2] for name, values in inputs.items()} for index in range(8)],
        batch_size=2,
    )
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "max_epochs": -1,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }

    class FixedStepMasks(L.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            torch.manual_seed(900 + trainer.global_step)

    first = model(True)
    trainer = L.Trainer(
        max_steps=2, callbacks=[FixedStepMasks()], default_root_dir=tmp_path, **options
    )
    trainer.fit(first, train_dataloaders=loader)
    path = tmp_path / "full-state.ckpt"
    trainer.save_checkpoint(path)
    state = torch.load(path, weights_only=True, map_location="cpu")

    class VerifyRestoredState(FixedStepMasks):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == int(pl_module.ema_updates) == 2
            assert_tree_close(pl_module.state_dict(), state["state_dict"], rtol=0, atol=0)
            assert_tree_close(
                trainer.optimizers[0].state_dict(), state["optimizer_states"][0], rtol=0, atol=0
            )
            assert_tree_close(
                trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                state["lr_schedulers"][0],
                rtol=0,
                atol=0,
            )

    outcomes = []
    for checkpointing in (True, False):
        restored = model(checkpointing)
        resumed = L.Trainer(
            max_steps=3,
            callbacks=[VerifyRestoredState()],
            default_root_dir=tmp_path / str(checkpointing),
            **options,
        )
        resumed_loader = DataLoader(loader.dataset, batch_size=resumed_batch)
        resumed.fit(
            restored, train_dataloaders=resumed_loader, ckpt_path=str(path), weights_only=True
        )
        assert resumed.global_step == int(restored.ema_updates) == 3
        assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 3
        assert restored._resolved_training_schedule == first._resolved_training_schedule
        outcomes.append(
            (deepcopy(restored.state_dict()), deepcopy(resumed.optimizers[0].state_dict()))
        )
    assert_tree_close(outcomes[0], outcomes[1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("devices", "mode", "expected"),
    [(8, "auto", False), (8, "on", True), (8, "off", False), (4, "auto", True), (4, "off", False)],
)
def test_segment_execution_choice_preserves_recipe(tmp_path, devices, mode, expected):
    plan = config_plan(tmp_path) | {
        "distributed": {"nnodes": 1, "nproc_per_node": devices},
        "activation_checkpointing": mode,
    }
    cfg = make_config(plan)
    assert cfg.model.backbone.config.use_gradient_checkpointing is expected
    assert cfg.data.batch_size * cfg.trainer.devices == 16
    assert cfg.data.masked_crops_per_recording == 8
    assert cfg.trainer.max_steps == cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.trainer.get("accumulate_grad_batches", 1) == 1
    assert cfg.model.backbone.config.compact_masked_tokens is True
    assert cfg.model.masked_weight == 1 and cfg.model.temporal_variance_weight == 0.1
    assert cfg.data.shard_prefetch == int(devices == 8)
    assert cfg.data.num_workers == 1


def test_operator_stop_precedes_durable_publication_and_is_not_model_metadata(tmp_path):
    from hydra.utils import instantiate

    control = {
        "request_path": str(tmp_path / "operator-stop.json"),
        "run_id": "test-training-run",
        "segment_id": "test-segment-001",
        "nonce": "b" * 32,
    }
    plan = config_plan(tmp_path) | {"operator_stop": control, "shard_prefetch": 1}
    baseline = make_config(config_plan(tmp_path))
    cfg = make_config(plan)
    order = list(cfg.callbacks)
    assert order.index("operator_stop") < order.index("model_checkpoint")
    assert cfg.model == baseline.model
    assert instantiate(cfg.callbacks.operator_stop).expected["nonce"] == control["nonce"]
    assert cfg.data.shard_prefetch == 1
    assert cfg.data.num_workers == baseline.data.num_workers == 1
    assert make_config(plan | {"shard_prefetch": 0}).data.shard_prefetch == 0

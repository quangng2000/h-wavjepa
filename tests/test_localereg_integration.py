"""End-to-end gradient, checkpoint and configuration contracts for LocaLeReg."""

import lightning as L
import pytest
import torch
from test_hierarchical_dynamics import checkpoint_for, gradient_magnitude, model_factory, waveform
from torch.utils.data import DataLoader

from src.dynamics_inference import load_model
from src.objectives.localereg import SOURCE_COMMIT, LocaLeReg


def test_all_three_local_losses_reach_the_trainable_backbone_and_abstractions():
    model = model_factory(regularizer_kind="localereg", localereg_clusters=2)
    audio = waveform(16)
    rng = torch.random.get_rng_state()
    result = model.compute_loss(audio, seed=53)
    assert torch.equal(rng, torch.random.get_rng_state())
    repeated = model.compute_loss(audio, seed=53)
    for actual, again in zip(
        result["regularization_terms"], repeated["regularization_terms"], strict=True
    ):
        torch.testing.assert_close(actual, again, rtol=0, atol=0)
    assert len(result["regularization_stats"]) == 3
    assert all(stats["effective_clusters"] >= 1 for stats in result["regularization_stats"])
    result["regularization_loss"].backward()
    assert gradient_magnitude(model.backbone.model.extract_audio) > 0
    assert gradient_magnitude(model.backbone.model.encoder) > 0
    assert all(gradient_magnitude(module) > 0 for module in model.abstractions)
    assert all(p.grad is None for p in model.teacher_abstractions.parameters())
    assert all(p.grad is None for p in model.backbone.model.teacher_encoder.parameters())


def test_local_checkpoint_loads_its_exact_regularizer_and_predictions(tmp_path):
    model = model_factory(regularizer_kind="localereg").eval()
    path = tmp_path / "local.ckpt"
    torch.save(checkpoint_for(model), path)
    restored = load_model(path)
    assert isinstance(restored.regularizer, LocaLeReg)
    assert restored.checkpoint_metadata() == model.checkpoint_metadata()
    assert restored.checkpoint_metadata()["localereg_commit"] == SOURCE_COMMIT
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    audio = waveform()[:, : model.context_windows * model.window_samples]
    with torch.no_grad():
        for before, after in zip(model.predict(audio), restored.predict(audio), strict=True):
            # Frozen inference may select a different floating-point kernel.
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "override",
    [{"regularizer_kind": "sigreg"}, {"localereg_clusters": 8}, {"localereg_n_quad": 64}],
)
def test_resume_rejects_changed_regularization(override):
    source = model_factory(regularizer_kind="localereg")
    restored = model_factory(**({"regularizer_kind": "localereg"} | override))
    with pytest.raises(ValueError, match="mismatch"):
        restored.on_load_checkpoint(checkpoint_for(source))


def test_local_checkpoint_rejects_changed_source():
    model = model_factory(regularizer_kind="localereg")
    checkpoint = checkpoint_for(model)
    checkpoint[model.CHECKPOINT_KEY]["localereg_commit"] = "unrecognized-revision"
    with pytest.raises(ValueError, match="source revision mismatch"):
        model.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "override",
    [
        {"regularizer_kind": "other"},
        {"localereg_clusters": True},
        {"localereg_min_cluster_size": 1},
        {"localereg_kmeans_iters": 0},
        {"localereg_n_quad": 1},
    ],
)
def test_invalid_regularizer_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        model_factory(**override)


def test_default_experiments_select_localereg_and_sigreg_control_is_explicit(make_config):
    for experiment in [
        "upstream_hierarchical",
        "upstream_flat",
        "upstream_pretrained",
        "upstream_smoke",
    ]:
        cfg = make_config([f"experiment={experiment}"])
        assert cfg.model.regularizer_kind == "localereg"
        assert cfg.model.localereg_n_quad == 256
        if experiment != "upstream_smoke":
            assert cfg.data.batch_size == 32
    control = make_config(["experiment=upstream_sigreg"])
    assert control.model.regularizer_kind == "sigreg"
    assert control.data.batch_size == 32


@pytest.mark.integration
def test_local_training_resume_updates_weights_and_logs_partition_diagnostics(tmp_path):
    torch.manual_seed(59)
    options = dict(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    model = model_factory(regularizer_kind="localereg", localereg_clusters=2)
    initial = model.abstractions[0][0].weight.detach().clone()
    loader = DataLoader([{"waveform": row} for row in waveform(16)], batch_size=16)
    trainer = L.Trainer(max_steps=1, **options)
    trainer.fit(model, train_dataloaders=loader)
    assert not torch.equal(initial, model.abstractions[0][0].weight)
    for level in range(3):
        assert trainer.callback_metrics[f"train/level_{level}/localereg_fallback"] == 0
        assert trainer.callback_metrics[f"train/level_{level}/localereg_effective_clusters"] >= 1
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)
    restored = model_factory(regularizer_kind="localereg", localereg_clusters=2)
    resumed = L.Trainer(max_steps=2, **options)
    resumed.fit(restored, train_dataloaders=loader, ckpt_path=str(path))
    assert restored.ema_updates.item() == resumed.global_step == 2
    assert not torch.equal(model.abstractions[0][0].weight, restored.abstractions[0][0].weight)

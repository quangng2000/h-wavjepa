"""Held-out controls expose temporal collapse without changing the objective."""

from unittest.mock import patch

import lightning as L
import pytest
import torch
from test_hierarchical_dynamics import checkpoint_for, model_factory, waveform
from torch.utils.data import DataLoader

from src.dynamics_inference import load_model
from src.objectives.temporal_diagnostics import forecast_diagnostics, recording_variance


def test_recording_constant_states_have_diversity_but_no_temporal_variance():
    states = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 5, 3)
    result = recording_variance(states)
    assert result["within_recording_variance"] == 0
    assert result["between_recording_variance"] > 0
    assert result["within_variance_fraction"] == 0
    stats = forecast_diagnostics(states[:, -1], states[:, -1], states[:, -1], states[:, 0])
    assert all(value == 0 for value in stats.values())
    # An ideal predictor and diverse batch statistics alone cannot rule this out.
    assert torch.nn.functional.mse_loss(states[:, -1], states[:, -1]) == 0


def test_global_collapse_and_temporal_variation_have_distinct_diagnostics():
    collapsed = recording_variance(torch.zeros(4, 5, 3))
    assert all(value == 0 for value in collapsed.values())
    states = torch.arange(5, dtype=torch.float32)[None, :, None].expand(4, 5, 3)
    temporal = recording_variance(states)
    assert temporal["within_recording_variance"] > 0
    assert temporal["between_recording_variance"] == 0
    assert temporal["within_variance_fraction"] == 1


def test_variance_decomposition_and_forecast_controls_have_known_values():
    states = torch.randn(4, 5, 6, requires_grad=True)
    stats = recording_variance(states)
    torch.testing.assert_close(
        stats["within_recording_variance"] + stats["between_recording_variance"],
        states.flatten(0, 1).var(0, unbiased=False).mean(),
    )
    assert all(not value.requires_grad for value in stats.values())
    target = torch.full((4, 6), 2.0)
    forecast = forecast_diagnostics(target, target, torch.ones_like(target), target * 2)
    assert forecast["persistence_mse"] == 1
    assert forecast["forecast_gain_over_persistence"] == 1
    assert forecast["time_shifted_context_mse"] == 4
    assert forecast["time_shifted_context_mse_increase"] == 4


@pytest.mark.parametrize("masked_weight", [0.0, 1.0])
def test_diagnostics_use_exact_teacher_origin_and_do_not_change_loss(masked_weight):
    torch.manual_seed(98)
    model = model_factory(masked_weight=masked_weight).eval()
    # Deliberately create student/teacher differences; student-origin persistence
    # would confuse EMA drift with temporal dynamics and fail this assertion.
    with torch.no_grad():
        for parameter in model.abstractions.parameters():
            parameter.add_(0.2)
    audio = waveform(4)
    baseline = model.compute_loss(audio, seed=17)
    rng = torch.get_rng_state()
    with patch.object(
        model.backbone, "encode_teacher_windows", wraps=model.backbone.encode_teacher_windows
    ) as teacher:
        result = model.compute_loss(audio, seed=17, diagnostics=True)
        assert teacher.call_count == (0 if masked_weight else 2)
    stats = model.temporal_diagnostics(result)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(baseline["loss"], result["loss"], rtol=0, atol=0)
    dense_targets = model.encode_targets(audio)
    shifted = model._predict_levels([level.roll(1, dims=1) for level in result["levels"]])
    for i in range(3):
        expected_origin = dense_targets[i][:, model.context_windows - 1]
        torch.testing.assert_close(result["teacher_origins"][i], expected_origin)
        torch.testing.assert_close(
            stats[i]["persistence_mse"],
            (expected_origin - result["targets"][i]).square().mean(),
        )
        torch.testing.assert_close(
            stats[i]["time_shifted_context_mse"],
            (shifted[i] - result["targets"][i]).square().mean(),
        )
        assert all(not value.requires_grad for value in stats[i].values())


@pytest.mark.parametrize("strides", [[1, 1, 1], [1, 2, 4]])
def test_flat_control_regularizes_shared_full_context_space_once(strides):
    model = model_factory(hierarchy=False, state_strides=strides).eval()
    audio = waveform(4)
    with patch.object(model.regularizer, "forward", wraps=model.regularizer.forward) as regularizer:
        result = model.compute_loss(audio, seed=31)
        assert regularizer.call_count == 1
        assert regularizer.call_args.args[0].shape[:2] == (model.context_windows, 4)
    assert result["regularization_terms"][1:] == [0, 0]
    assert result["regularization_loss"] == result["regularization_terms"][0]
    # Predictor horizons and stride grids remain independent controls.
    assert len(result["prediction_terms"]) == 3
    assert [level.shape[1] for level in result["levels"]] == [
        (model.context_windows - 1) // stride + 1 for stride in strides
    ]


def test_hierarchical_regularization_retains_three_distinct_space_penalties():
    model = model_factory().eval()
    with patch.object(model.regularizer, "forward", wraps=model.regularizer.forward) as regularizer:
        result = model.compute_loss(waveform(4), seed=31)
    assert regularizer.call_count == 3
    torch.testing.assert_close(result["regularization_loss"], sum(result["regularization_terms"]))


def test_old_flat_checkpoint_keeps_per_level_objective_and_requires_explicit_migration(tmp_path):
    legacy = model_factory(hierarchy=False, regularization_scope="per_level").eval()
    checkpoint = checkpoint_for(legacy)
    checkpoint[legacy.CHECKPOINT_KEY]["dynamics"].pop("regularization_scope")
    path = tmp_path / "legacy-flat.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.regularization_scope == "per_level"
    with patch.object(
        restored.regularizer, "forward", wraps=restored.regularizer.forward
    ) as regularizer:
        restored.compute_loss(waveform(4), seed=31)
    assert regularizer.call_count == 3
    with pytest.raises(ValueError, match="configuration mismatch"):
        model_factory(hierarchy=False).on_load_checkpoint(checkpoint)


@pytest.mark.integration
def test_validation_logs_each_level_temporal_diagnostics(tmp_path):
    model = model_factory()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader([{"waveform": row} for row in waveform(4)], batch_size=4)
    trainer.validate(model, dataloaders=loader)
    for i in range(3):
        for name in (
            "within_recording_variance",
            "between_recording_variance",
            "persistence_mse",
            "time_shifted_context_mse_increase",
        ):
            assert torch.isfinite(trainer.callback_metrics[f"val/level_{i}/{name}"])

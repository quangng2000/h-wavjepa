"""Causal support and within-recording pressure for the temporal audio variant."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from test_hierarchical_dynamics import (
    WINDOW_SAMPLES,
    checkpoint_for,
    gradient_magnitude,
    model_factory,
    waveform,
)
from torch import nn

from src.dynamics_inference import load_model
from src.objectives.temporal_variance import temporal_variance_floor


def temporal_model(**overrides):
    return model_factory(**({"aggregation_windows": [1, 2, 2]} | overrides))


def test_dense_valid_support_is_compositional_and_sparse_positions_are_physical_endpoints():
    model = temporal_model(hierarchy=False)
    low = torch.arange(9, dtype=torch.float32).square()[None, :, None].expand(2, 9, 16)
    dense = model.levels_from_low(low)
    sparse = model.levels_from_low(low, subsample=True)
    assert model.level_support_windows == (1, 2, 4)
    assert [p.tolist() for p in model.level_positions(5)] == [[0, 1, 2, 3, 4], [1, 2, 3, 4], [3, 4]]
    assert [p.tolist() for p in model.level_positions(5, subsample=True)] == [
        [0, 1, 2, 3, 4],
        [2, 4],
        [4],
    ]
    for level, support, positions in zip(dense, (1, 2, 4), model.level_positions(9), strict=True):
        expected = torch.stack(
            [low[:, endpoint - support + 1 : endpoint + 1].mean(1) for endpoint in positions],
            dim=1,
        )
        torch.testing.assert_close(level, expected)
    for full, sampled, positions, support in zip(
        dense, sparse, model.level_positions(9, subsample=True), (1, 2, 4), strict=True
    ):
        torch.testing.assert_close(sampled, full[:, positions - support + 1], rtol=0, atol=0)


def test_upper_aggregation_precedes_nonlinear_map_and_uses_lower_stride():
    model = temporal_model()

    class Square(nn.Module):
        def forward(self, values):
            return values.square()

    model.abstractions = nn.ModuleList([Square(), Square()])
    low = torch.arange(7, dtype=torch.float32)[None, :, None]
    _, medium, high = model.levels_from_low(low)
    expected_medium = ((low[:, :-1] + low[:, 1:]) / 2).square()
    expected_high = ((expected_medium[:, :-2] + expected_medium[:, 2:]) / 2).square()
    torch.testing.assert_close(medium, expected_medium)
    torch.testing.assert_close(high, expected_high)


def test_high_state_uses_every_sample_in_its_causal_support_and_nothing_else():
    model = temporal_model(hierarchy=False)
    low = torch.randn(2, 9, 16, requires_grad=True)
    model.levels_from_low(low)[-1][:, 1].sum().backward()  # endpoint 4, support 1..4
    torch.testing.assert_close(low.grad[:, 1:5], torch.full_like(low.grad[:, 1:5], 0.25))
    assert torch.count_nonzero(low.grad[:, [0, 5, 6, 7, 8]]) == 0
    changed = low.detach().clone()
    changed[:, 5:] += 100
    for before, after in zip(
        model.levels_from_low(low.detach()), model.levels_from_low(changed), strict=True
    ):
        support = 9 - before.shape[1] + 1
        torch.testing.assert_close(
            before[:, : 6 - support], after[:, : 6 - support], rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "windows,counts", [(1, [1, 0, 0]), (2, [2, 1, 0]), (3, [3, 2, 0]), (4, [4, 3, 1])]
)
def test_short_inputs_return_empty_upper_levels_without_padding(windows, counts):
    model = temporal_model().eval()
    low = torch.randn(2, windows, 16)
    assert [values.shape[1] for values in model.levels_from_low(low)] == counts
    assert [len(values) for values in model.level_positions(windows)] == counts
    audio = torch.randn(2, windows * WINDOW_SAMPLES)
    assert [values.shape[1] for values in model.encode(audio)] == counts


@pytest.mark.parametrize("masked_weight", [0.0, 1.0])
def test_teacher_targets_and_origins_use_complete_support_and_share_dense_teacher(masked_weight):
    model = temporal_model(masked_weight=masked_weight).eval()
    audio = waveform()
    with patch.object(
        model.backbone, "encode_teacher_windows", wraps=model.backbone.encode_teacher_windows
    ) as teacher:
        result = model.compute_loss(audio, seed=71, diagnostics=True)
    assert teacher.call_count == (0 if masked_weight else 1)
    if not masked_weight:
        assert teacher.call_args.args[0].shape[1] == 9
    dense_teacher = model.encode_targets(audio)
    for i, (horizon, support) in enumerate(
        zip(model.horizons, model.level_support_windows, strict=True)
    ):
        torch.testing.assert_close(result["targets"][i], dense_teacher[i][:, 5 - support + horizon])
        torch.testing.assert_close(result["teacher_origins"][i], dense_teacher[i][:, 5 - support])
    stats = model.temporal_diagnostics(result)
    assert [stat["variance_context_states"].item() for stat in stats] == [5, 4, 2]
    assert [stat["predictor_context_states"].item() for stat in stats] == [5, 2, 1]
    assert stats[-1]["time_shifted_context_mse_increase"] == 0


def test_aggregated_predictions_ignore_future_audio_and_keep_absolute_context_positions():
    model = temporal_model(temporal_variance_weight=0.2).eval()
    audio = waveform().requires_grad_()
    changed = audio.detach().clone()
    changed[:, 5 * WINDOW_SAMPLES :] += 100
    result = model.compute_loss(audio, seed=7)
    altered = model.compute_loss(changed, seed=7)
    observed = []
    handles = [
        predictor.register_forward_pre_hook(lambda _module, args: observed.append(args))
        for predictor in model.predictors
    ]
    try:
        model.predict(audio[:, : 5 * WINDOW_SAMPLES])
    finally:
        for handle in handles:
            handle.remove()
    assert [positions.tolist() for _, positions in observed] == [[0, 1, 2, 3, 4], [2, 4], [4]]
    for before, after in zip(result["predictions"], altered["predictions"], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(
        result["temporal_variance_loss"], altered["temporal_variance_loss"], rtol=0, atol=0
    )
    result["predictions"][-1].square().sum().backward()
    assert audio.grad[:, : 5 * WINDOW_SAMPLES].abs().sum() > 0
    assert torch.count_nonzero(audio.grad[:, 5 * WINDOW_SAMPLES :]) == 0
    assert gradient_magnitude(model.abstractions[0]) > 0
    assert gradient_magnitude(model.abstractions[1]) > 0
    assert all(p.grad is None for p in model.teacher_abstractions.parameters())


def test_recording_identity_diversity_cannot_satisfy_temporal_floor():
    identities = torch.arange(6, dtype=torch.float32)[:, None, None].expand(6, 5, 4)
    penalty = temporal_variance_floor(identities)
    torch.testing.assert_close(penalty, torch.tensor(0.09))
    # Strong variation across recordings and their arbitrary offsets is irrelevant.
    torch.testing.assert_close(temporal_variance_floor(identities * 100), penalty)
    changing = identities + torch.arange(5, dtype=torch.float32)[None, :, None]
    assert temporal_variance_floor(changing) == 0


def test_floor_is_per_recording_per_feature_and_has_finite_restoring_gradient():
    states = torch.tensor(
        [[[0.0, 10.0], [0.02, 10.01]], [[3.0, 20.0], [3.01, 20.02]]], requires_grad=True
    )
    penalty = temporal_variance_floor(states)
    expected = torch.relu(0.1 - torch.sqrt(states.float().var(1, unbiased=False) + 1e-4)).mean()
    torch.testing.assert_close(penalty, expected)
    penalty.backward()
    assert torch.isfinite(states.grad).all()
    assert states.grad.abs().sum() > 0
    assert torch.all(states.grad[:, 0] > 0)
    assert torch.all(states.grad[:, 1] < 0)
    assert temporal_variance_floor(states.to(torch.bfloat16)).dtype == torch.float32


def test_temporal_penalty_uses_dense_online_prefix_and_reaches_backbone_and_both_maps():
    model = temporal_model(temporal_variance_weight=0.3, temporal_std_floor=2.0)
    result = model.compute_loss(waveform(), seed=21)
    assert result["temporal_variance_terms"][0] == 0
    for i in (1, 2):
        manual = temporal_variance_floor(result["dense_levels"][i], std_floor=2.0)
        torch.testing.assert_close(result["temporal_variance_terms"][i], manual)
    torch.testing.assert_close(
        result["loss"],
        result["masked_loss"]
        + result["prediction_loss"]
        + 0.1 * result["regularization_loss"]
        + 0.3 * result["temporal_variance_loss"],
    )
    result["temporal_variance_loss"].backward()
    assert gradient_magnitude(model.abstractions[0]) > 0
    assert gradient_magnitude(model.abstractions[1]) > 0
    assert gradient_magnitude(model.backbone.model.encoder) > 0
    assert all(parameter.grad is None for parameter in model.teacher_abstractions.parameters())


def test_temporal_penalty_participates_in_nonfinite_loss_guard():
    model = temporal_model(temporal_variance_weight=0.3)
    with (
        patch(
            "src.models.hierarchical_wavjepa_module.temporal_variance_floor",
            return_value=torch.tensor(float("nan")),
        ),
        pytest.raises(FloatingPointError, match="non-finite"),
    ):
        model.compute_loss(waveform(), seed=21)


def test_initial_teacher_and_online_temporal_states_share_normalized_readout():
    model = temporal_model(readout_mode="normalized_top_k").eval()
    audio = waveform()
    for online, teacher in zip(model.encode(audio), model.encode_targets(audio), strict=True):
        torch.testing.assert_close(online, teacher, rtol=1e-5, atol=1e-6)
        assert online.requires_grad
        assert not teacher.requires_grad


def test_pointwise_temporal_pressure_can_be_ablated_without_temporal_pooling():
    model = model_factory(temporal_variance_weight=0.2).eval()
    result = model.compute_loss(waveform(), seed=47)
    assert model.level_support_windows == (1, 1, 1)
    assert [level.shape[1] for level in result["dense_levels"]] == [5, 5, 5]
    assert result["temporal_variance_terms"][1] > 0
    assert result["temporal_variance_terms"][2] > 0


@pytest.mark.parametrize("aggregation,counts", [([1, 1, 1], [5, 5, 5]), ([1, 2, 2], [5, 4, 2])])
def test_diagnostics_use_dense_grid_without_changing_loss_or_gradients(aggregation, counts):
    ordinary = model_factory(aggregation_windows=aggregation).eval()
    diagnostic = deepcopy(ordinary)
    audio = waveform()
    before = ordinary.compute_loss(audio, seed=71)
    after = diagnostic.compute_loss(audio, seed=71, diagnostics=True)
    stats = diagnostic.temporal_diagnostics(after)
    assert [stat["variance_context_states"].item() for stat in stats] == counts
    torch.testing.assert_close(before["loss"], after["loss"], rtol=0, atol=0)
    before["loss"].backward()
    after["loss"].backward()
    for (name, left), (_, right) in zip(
        ordinary.named_parameters(), diagnostic.named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)


def test_zeroed_upper_maps_have_positive_temporal_penalty_despite_zero_forecasting_error():
    model = temporal_model(temporal_variance_weight=0.3).eval()
    with torch.no_grad():
        for maps in (model.abstractions, model.teacher_abstractions, model.predictors):
            for parameter in maps.parameters():
                parameter.zero_()
    result = model.compute_loss(waveform(), seed=21)
    assert all(term == 0 for term in result["prediction_terms"][1:])
    for term in result["temporal_variance_terms"][1:]:
        torch.testing.assert_close(term, torch.tensor(0.09))


def test_pointwise_baseline_retains_original_composition_and_optional_penalty_is_separate():
    model = model_factory().eval()
    low = torch.randn(3, 9, 16)
    for subsample in (False, True):
        before = model._levels(
            low, model.abstractions, model.state_strides if subsample else (1, 1, 1)
        )
        after = model.levels_from_low(low, subsample=subsample)
        for left, right in zip(before, after, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    audio = waveform()
    result = model.compute_loss(audio, seed=31)
    assert result["temporal_variance_loss"] == 0
    torch.testing.assert_close(
        result["loss"],
        result["masked_loss"] + result["prediction_loss"] + 0.1 * result["regularization_loss"],
        rtol=0,
        atol=0,
    )


def test_pooled_identity_control_regularizes_each_distinct_pooled_space():
    model = temporal_model(hierarchy=False).eval()
    with patch.object(model.regularizer, "forward", wraps=model.regularizer.forward) as regularizer:
        result = model.compute_loss(waveform(), seed=31)
    assert regularizer.call_count == 3
    assert all(torch.isfinite(term) for term in result["regularization_terms"])


@pytest.mark.parametrize(
    "options",
    [
        {"aggregation_windows": [2, 2, 2]},
        {"aggregation_windows": [1, 0, 2]},
        {"aggregation_windows": [1, True, 2]},
        {"aggregation_windows": [1, 2]},
        {"aggregation_windows": [1, 3, 3]},
        {"temporal_variance_weight": -1},
        {"temporal_variance_weight": float("nan")},
        {"temporal_variance_weight": True},
        {"temporal_variance_weight": "0.1"},
        {"temporal_variance_epsilon": 0},
        {"temporal_variance_epsilon": True},
        {"temporal_variance_epsilon": None},
        {"temporal_variance_epsilon": float("inf")},
        {"temporal_std_floor": 0.01},
        {"temporal_std_floor": True},
        {"temporal_std_floor": "0.1"},
        {"temporal_std_floor": float("nan")},
        {"temporal_variance_weight": 0.1, "hierarchy": False},
        {"temporal_variance_weight": 0.1, "prediction_weights": [1, 0, 1]},
        {"temporal_variance_weight": 0.1, "aggregation_windows": [1, 3, 2]},
    ],
)
def test_invalid_aggregation_and_temporal_variance_contracts_fail_early(options):
    with pytest.raises(ValueError):
        temporal_model(**options)


def test_checkpoint_roundtrip_records_aggregation_and_penalty_and_rejects_changed_semantics(
    tmp_path,
):
    model = temporal_model(temporal_variance_weight=0.3).eval()
    checkpoint = checkpoint_for(model)
    path = tmp_path / "temporal.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.aggregation_windows == (1, 2, 2)
    assert restored.temporal_variance_weight == 0.3
    audio = waveform()
    torch.testing.assert_close(
        model.compute_loss(audio, seed=11)["loss"],
        restored.compute_loss(audio, seed=11)["loss"],
        rtol=0,
        atol=0,
    )
    for changed in (
        model_factory(),
        temporal_model(),
        temporal_model(temporal_variance_weight=0.2),
    ):
        with pytest.raises(ValueError, match="configuration mismatch"):
            changed.on_load_checkpoint(checkpoint)


def test_old_checkpoint_defaults_to_original_pointwise_objective(tmp_path):
    source = model_factory().eval()
    checkpoint = checkpoint_for(source)
    for field in (
        "aggregation_windows",
        "temporal_variance_weight",
        "temporal_std_floor",
        "temporal_variance_epsilon",
    ):
        checkpoint[source.CHECKPOINT_KEY]["dynamics"].pop(field)
    path = tmp_path / "old-pointwise.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.aggregation_windows == (1, 1, 1)
    assert restored.temporal_variance_weight == 0
    source.on_load_checkpoint(checkpoint)
    audio = waveform()
    torch.testing.assert_close(
        source.compute_loss(audio, seed=31)["loss"],
        restored.compute_loss(audio, seed=31)["loss"],
        rtol=0,
        atol=0,
    )

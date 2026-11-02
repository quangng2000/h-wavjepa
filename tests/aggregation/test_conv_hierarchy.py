"""Convolution changes aggregation while preserving forecast causality."""

import torch
from test_hierarchical_dynamics import WINDOW_SAMPLES, gradient_magnitude, model_factory, waveform


def conv_model(**overrides):
    return model_factory(
        **({"aggregation_mode": "conv1d", "aggregation_windows": [1, 2, 2]} | overrides)
    )


def test_mean_initialized_convolution_matches_mean_hierarchy():
    torch.manual_seed(43)
    baseline = model_factory(aggregation_windows=[1, 2, 2]).eval()
    torch.manual_seed(43)
    candidate = conv_model().eval()
    values = torch.randn(2, 9, 16)
    for left, right in zip(
        baseline.levels_from_low(values), candidate.levels_from_low(values), strict=True
    ):
        torch.testing.assert_close(left, right)
    for left, right in zip(
        baseline.predictors.parameters(), candidate.predictors.parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    audio = waveform()
    torch.testing.assert_close(
        baseline.compute_loss(audio, seed=41)["loss"],
        candidate.compute_loss(audio, seed=41)["loss"],
    )


def test_dense_and_strided_states_have_the_same_physical_support():
    model = conv_model().eval()
    values = torch.randn(2, 9, 16, requires_grad=True)
    dense = model.levels_from_low(values)
    sparse = model.levels_from_low(values, subsample=True)
    assert model.level_support_windows == (1, 2, 4)
    assert [level.shape[1] for level in dense] == [9, 8, 6]
    assert [level.shape[1] for level in sparse] == [9, 4, 2]
    for full, sampled, support, positions in zip(
        dense,
        sparse,
        model.level_support_windows,
        model.level_positions(9, subsample=True),
        strict=True,
    ):
        torch.testing.assert_close(sampled, full[:, positions - support + 1])
    dense[-1][:, 1].square().sum().backward()  # Endpoint 4 has support 1..4.
    assert values.grad[:, 1:5].abs().sum() > 0
    assert torch.count_nonzero(values.grad[:, [0, 5, 6, 7, 8]]) == 0


def test_future_audio_cannot_change_predictions_and_targets_remain_aligned():
    model = conv_model().eval()
    audio = waveform()
    changed = audio.clone()
    changed[:, 5 * WINDOW_SAMPLES :] += 100
    original = model.compute_loss(audio, seed=31)
    altered = model.compute_loss(changed, seed=31)
    teacher = model.encode_targets(audio)
    for index, (before, after) in enumerate(
        zip(original["predictions"], altered["predictions"], strict=True)
    ):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        target = teacher[index][:, 5 + model.horizons[index] - model.level_support_windows[index]]
        torch.testing.assert_close(original["targets"][index], target)


def test_forecasting_trains_both_convolutions_and_backbone_but_not_teacher():
    model = conv_model()
    loss = model.compute_loss(waveform(), seed=23)["prediction_terms"][-1]
    loss.backward()
    for abstraction in model.abstractions:
        assert gradient_magnitude(abstraction[0]) > 0
        assert all(torch.isfinite(p.grad).all() for p in abstraction[0].parameters())
    assert gradient_magnitude(model.backbone.model.encoder) > 0
    assert all(p.grad is None for p in model.teacher_abstractions.parameters())
    assert all(not p.requires_grad for p in model.teacher_abstractions.parameters())

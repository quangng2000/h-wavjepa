"""Other valid kernel lengths retain physical positions and target indices."""

import pytest
import torch
from test_hierarchical_dynamics import WINDOW_SAMPLES, model_factory


@pytest.mark.parametrize(
    "kernels,context,horizons,supports",
    [
        ([1, 1, 1], 5, [1, 2, 4], (1, 1, 1)),
        ([1, 3, 3], 9, [1, 4, 8], (1, 3, 7)),
    ],
)
def test_valid_geometry_retains_targets_and_dense_sparse_alignment(
    kernels,
    context,
    horizons,
    supports,
):
    model = model_factory(
        aggregation_mode="conv1d",
        aggregation_windows=kernels,
        context_windows=context,
        horizons=horizons,
    ).eval()
    count = context + max(horizons)
    audio = torch.randn(2, count * WINDOW_SAMPLES)
    result = model.compute_loss(audio, seed=23)
    targets = model.encode_targets(audio)
    assert model.level_support_windows == supports
    assert torch.isfinite(result["loss"])
    for index, support in enumerate(supports):
        endpoint = context - 1 + horizons[index]
        torch.testing.assert_close(
            result["targets"][index], targets[index][:, endpoint - support + 1]
        )
    dense = model.encode(audio)
    sparse = model.encode(audio, subsample=True)
    for level, positions, support in zip(
        range(3), model.level_positions(count, subsample=True), supports, strict=True
    ):
        torch.testing.assert_close(sparse[level], dense[level][:, positions - support + 1])


@pytest.mark.parametrize(
    "options",
    [
        {"aggregation_mode": "unknown"},
        {"aggregation_mode": True},
        {"aggregation_mode": "conv1d", "hierarchy": False},
        {"aggregation_mode": "conv1d", "aggregation_windows": [1, 3, 3]},
    ],
)
def test_invalid_convolution_or_insufficient_context_fails_before_training(options):
    with pytest.raises(ValueError):
        model_factory(**options)

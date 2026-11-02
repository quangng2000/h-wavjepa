"""Valid convolution retains causal support and learns channel mixing."""

import pytest
import torch

from src.models.components.aggregation import CausalConv1d


def test_initialization_matches_dilated_mean_without_consuming_rng():
    torch.manual_seed(17)
    before = torch.random.get_rng_state()
    convolution = CausalConv1d(2, 3, 2)
    assert torch.equal(torch.random.get_rng_state(), before)
    values = torch.arange(18, dtype=torch.float32).reshape(1, 9, 2)
    expected = (values[:, :-4] + values[:, 2:-2] + values[:, 4:]) / 3
    torch.testing.assert_close(convolution(values), expected)


def test_learned_taps_mix_channels_in_oldest_to_current_order():
    convolution = CausalConv1d(2, 3, 2)
    weights = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[-1.0, 0.0, 2.0], [3.0, -2.0, 1.0]]]
    )
    with torch.no_grad():
        next(convolution.parameters()).copy_(weights)
    values = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    expected = []
    for endpoint in range(4, 8):
        window = values[:, [endpoint - 4, endpoint - 2, endpoint]].transpose(1, 2)
        expected.append(torch.einsum("bit,oit->bo", window, weights))
    torch.testing.assert_close(convolution(values), torch.stack(expected, dim=1))


def test_gradient_uses_only_completed_causal_taps():
    convolution = CausalConv1d(2, 3, 2)
    values = torch.randn(1, 9, 2, requires_grad=True)
    convolution(values)[:, 1].sum().backward()  # Endpoint 5: windows 1, 3, 5.
    expected = torch.zeros_like(values)
    expected[:, [1, 3, 5]] = 1 / 3
    torch.testing.assert_close(values.grad, expected)
    assert next(convolution.parameters()).grad.abs().sum() > 0


@pytest.mark.parametrize("length,count", [(0, 0), (2, 0), (4, 0), (5, 1), (9, 5)])
def test_short_sequences_omit_incomplete_prefix(length, count):
    convolution = CausalConv1d(2, 3, 2)
    assert convolution(torch.randn(3, length, 2)).shape == (3, count, 2)

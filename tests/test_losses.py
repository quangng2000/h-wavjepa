import pytest
import torch

from src.objectives.losses import masked_prediction_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_reduction_matches_indexed_loss_and_gradients(dtype):
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(2, 5, 3, generator=generator, dtype=dtype)
    prediction = values.clone().requires_grad_()
    reference = values.clone().requires_grad_()
    target = torch.randn(2, 5, 3, generator=generator, dtype=dtype).requires_grad_()
    mask = torch.tensor([[True, False, True, False, False], [False, True, True, True, False]])
    loss, _ = masked_prediction_loss([prediction], [target], [mask], [1.0])
    expected = (reference - target.detach()).square().mean(-1)[mask].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(prediction.grad, reference.grad)
    assert target.grad is None
    assert torch.count_nonzero(prediction.grad[~mask]) == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_masked_loss_has_deterministic_backward():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        prediction = torch.ones(2, 4, 3, device="mps", requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[True, False, True, False]] * 2, device="mps")
        loss, _ = masked_prediction_loss([prediction], [target], [mask], [1.0])
        loss.backward()
        assert torch.isfinite(prediction.grad).all()
        assert loss.item() == pytest.approx(1.0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)

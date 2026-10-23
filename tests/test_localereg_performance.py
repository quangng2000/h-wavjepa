"""Reference parity for the optional fixed-shape detached clustering adapter."""

import pytest
import torch

from src.objectives.localereg import LocaLeReg
from src.objectives.localereg_fast import fixed_shape_kmeans_labels
from src.vendor.localereg.losses import _torch_kmeans_labels


@pytest.mark.parametrize("population", ["random", "collapsed", "duplicates", "zeros"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_shape_assignments_match_upstream_exactly(population, normalize, dtype):
    torch.manual_seed(31)
    states = torch.randn(64, 32).to(dtype)
    if population == "collapsed":
        states[:] = states[0].clone()
    elif population == "duplicates":
        states[32:] = states[0].clone()
    elif population == "zeros":
        states.zero_()
    torch.manual_seed(41)
    expected = _torch_kmeans_labels(states, 4, iters=10, normalize=normalize)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(41)
    actual = fixed_shape_kmeans_labels(states, 4, iters=10, normalize=normalize)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    if population in {"collapsed", "zeros"}:
        # Three empty clusters exercise preserving the previous centroid.
        assert actual.unique().numel() == 1


@pytest.mark.parametrize("population", ["random", "collapsed", "fallback", "duplicates"])
def test_fixed_shape_localereg_matches_values_gradients_diagnostics_and_rng(population):
    torch.manual_seed(43)
    states = torch.randn(3, 2 if population == "fallback" else 64, 16)
    if population == "collapsed":
        states[:] = states[:, :1].clone()
    elif population == "duplicates":
        states[:, 32:] = states[:, :1].clone()
    states.requires_grad_()
    reference = states.detach().clone().requires_grad_()
    kwargs = dict(num_proj=8, num_clusters=4, min_cluster_size=4, kmeans_iters=10, n_quad=32)
    official = LocaLeReg(**kwargs)
    candidate = LocaLeReg(**kwargs, fixed_shape_kmeans=True)
    torch.manual_seed(47)
    expected = official(reference)
    expected.backward()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(47)
    actual = candidate(states)
    actual.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(states.grad, reference.grad, rtol=0, atol=0)
    assert candidate.last_stats == official.last_stats
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for kernel parity")
def test_cuda_fixed_shape_localereg_matches_reference():
    torch.manual_seed(53)
    original = torch.randn(3, 64, 768, device="cuda", requires_grad=True)
    reference = original.detach().clone().requires_grad_()
    for states in (original, reference):
        with torch.no_grad():
            states[1:] = states[1:, :1].clone()
    kwargs = dict(num_proj=128, num_clusters=4, min_cluster_size=4, kmeans_iters=10, n_quad=256)
    official = LocaLeReg(**kwargs)
    candidate = LocaLeReg(**kwargs, fixed_shape_kmeans=True)
    expected = official(reference, seed=59)
    actual = candidate(original, seed=59)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(original.grad, reference.grad)
    assert candidate.last_stats == official.last_stats

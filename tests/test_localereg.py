"""Numerical checks for the pinned LocaLeReg code and its audio adaptation."""

import hashlib
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from src.objectives.localereg import SOURCE_COMMIT, LocaLeReg
from src.vendor.localereg import losses as upstream


def test_vendored_loss_is_byte_exact_at_the_pinned_revision():
    folder = Path(upstream.__file__).parent
    metadata = json.loads((folder / "UPSTREAM.json").read_text())
    source = (folder / "losses.py").read_bytes()
    assert metadata["commit"] == SOURCE_COMMIT
    assert hashlib.sha256(source).hexdigest() == metadata["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
        == (metadata["git_blob_sha1"])
    )
    assert metadata["git_blob_sha1"] == "d0fd0b71c6362278da80bbc926421cab4fa972b7"


def test_timewise_adapter_matches_official_values_and_gradients():
    torch.manual_seed(37)
    states = torch.randn(3, 32, 16, requires_grad=True)
    reference = states.detach().clone().requires_grad_()
    regularizer = LocaLeReg(num_proj=8)
    torch.manual_seed(41)
    actual = regularizer(states)
    actual.backward()
    torch.manual_seed(41)
    expected = torch.stack(
        [
            upstream.local_ballreg_loss(
                time.unsqueeze(0),
                loss=partial(upstream.ballreg_loss, center=True, num_slices=8, n_quad=256),
                num_clusters=4,
                min_cluster_size=4,
                kmeans_iters=10,
                weighted=False,
                normalize_for_kmeans=True,
                kmeans_backend="torch",
            )
            for time in reference
        ]
    ).mean()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(states.grad, reference.grad, rtol=0, atol=0)
    assert states.grad.abs().sum() > 0
    assert regularizer.last_stats["effective_clusters"] > 1
    assert regularizer.last_stats["fallback"] == 0


def test_fixed_cluster_centering_preserves_loss_and_gradient_under_translation():
    torch.manual_seed(43)
    states = torch.randn(1, 12, 8, requires_grad=True)
    translated = (states.detach() + torch.randn(1, 1, 8)).requires_grad_()
    torch.manual_seed(47)
    original = upstream.ballreg_loss(states, center=True, num_slices=32)
    original.backward()
    torch.manual_seed(47)
    shifted = upstream.ballreg_loss(translated, center=True, num_slices=32)
    shifted.backward()
    torch.testing.assert_close(original, shifted)
    torch.testing.assert_close(states.grad, translated.grad, atol=1e-7, rtol=1e-4)
    torch.testing.assert_close(states.grad.sum(dim=1), torch.zeros(1, 8), atol=1e-7, rtol=0)


def test_cluster_filtering_and_equal_cluster_weighting(monkeypatch):
    # Three retained neighborhoods with unequal sizes, plus one skipped singleton.
    labels = torch.tensor([0] * 4 + [1] * 5 + [2] * 6 + [3])
    monkeypatch.setattr(upstream, "_torch_kmeans_labels", lambda *args, **kwargs: labels)
    states = torch.arange(16.0).reshape(1, 16, 1).requires_grad_()
    stats = {}
    result = upstream.local_ballreg_loss(
        states,
        loss=lambda points: points.square().mean(),
        num_clusters=4,
        min_cluster_size=4,
        kmeans_backend="torch",
        stats=stats,
    )
    expected = torch.stack([states[:, labels == i].square().mean() for i in range(3)]).mean()
    torch.testing.assert_close(result, expected)
    result.backward()
    assert states.grad[0, -1, 0] == 0
    assert stats == dict(
        configured_clusters=4.0, effective_clusters=3.0, kept_fraction=15 / 16, fallback=0.0
    )


def test_small_batch_reports_global_fallback_and_collapsed_states_are_penalized():
    regularizer = LocaLeReg(num_proj=8)
    penalty = regularizer(torch.zeros(2, 2, 16, requires_grad=True))
    assert penalty > 0
    assert regularizer.last_stats == dict(
        configured_clusters=4.0, effective_clusters=0.0, kept_fraction=0.0, fallback=1.0
    )
    # A positive collapse penalty alone is not a guarantee of an escaping gradient.


def test_ball_characteristic_function_matches_the_analytic_interval_case():
    t = torch.linspace(0, 3, 17)
    actual = upstream._ball_d_cf(t, d=1, n_quad=256)
    expected = torch.sinc(math.sqrt(3) * t / math.pi)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)


def test_768_dimensional_target_matches_independent_float64_quadrature():
    t = torch.linspace(0, 3, 17)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    density = (1 - nodes**2) ** ((768 - 1) / 2) * weights
    reference = (
        np.cos(math.sqrt(770) * t.numpy().astype(np.float64)[:, None] * nodes)
        @ density
        / density.sum()
    )
    actual = upstream._ball_d_cf(t, d=768, n_quad=256)
    np.testing.assert_allclose(actual.numpy(), reference, atol=5e-6, rtol=0)
    # The upstream function default is under-resolved at our backbone width.
    assert np.max(np.abs(upstream._ball_d_cf(t, d=768).numpy() - reference)) > 0.2


@pytest.mark.parametrize(
    "states",
    [
        torch.randn(2, 1, 8),
        torch.randn(2, 8),
        torch.empty(0, 2, 8),
        torch.full((2, 2, 8), float("nan")),
    ],
)
def test_invalid_populations_are_rejected(states):
    with pytest.raises(ValueError, match="finite"):
        LocaLeReg()(states)


def test_unvalidated_precision_and_distributed_execution_are_rejected(monkeypatch):
    states = torch.randn(2, 4, 8)
    with pytest.raises(ValueError, match="float32"):
        LocaLeReg()(states.double())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="Distributed"):
        LocaLeReg()(states)

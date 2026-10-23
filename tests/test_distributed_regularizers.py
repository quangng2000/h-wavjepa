"""Cross-rank statistics and DDP gradients equal one independent global-batch loss."""

import importlib.util
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.objectives.localereg import LocaLeReg
from src.objectives.sigreg import SIGReg
from src.vendor.localereg.losses import ballreg_loss, local_ballreg_loss


def _regularizer(kind):
    if kind == "sigreg":
        return SIGReg(num_proj=8)
    return LocaLeReg(
        num_proj=8,
        num_clusters=2,
        min_cluster_size=2,
        n_quad=32,
        fixed_shape_kmeans=kind == "localereg_fixed_shape",
    )


def _reference(kind, states):
    if kind == "sigreg":
        path = Path(__file__).parent / "fixtures" / "h_jepa" / "loss.py"
        spec = importlib.util.spec_from_file_location("independent_sigreg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The independent reference sees the complete batch on one logical device;
        # its own upstream distributed reduction must not run a second time.
        with patch("torch.distributed.is_initialized", return_value=False):
            return module.SIGReg(num_proj=8)(states)
    return torch.stack(
        [
            local_ballreg_loss(
                time.unsqueeze(0),
                loss=partial(ballreg_loss, center=True, num_slices=8, n_quad=32),
                num_clusters=2,
                min_cluster_size=2,
                kmeans_iters=10,
                weighted=False,
                normalize_for_kmeans=True,
                kmeans_backend="torch",
            )
            for time in states
        ]
    ).mean()


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=45)
    )
    try:
        for kind in ("sigreg", "localereg", "localereg_fixed_shape"):
            torch.manual_seed(73)
            inputs = torch.randn(2, 12, 4)
            online = DistributedDataParallel(nn.Linear(4, 6))
            reference_model = nn.Linear(4, 6)
            reference_model.load_state_dict(online.module.state_dict())
            regularizer = _regularizer(kind)
            # Each rank enters with different RNG state and seed. Rank zero chooses
            # the sketch/partition; the regularizer must restore both RNG streams.
            torch.manual_seed(101 + rank)
            before = torch.get_rng_state().clone()
            actual = regularizer(online(inputs[:, rank * 6 : (rank + 1) * 6]), seed=509 + rank)
            assert torch.equal(before, torch.get_rng_state())
            actual.backward()
            with torch.random.fork_rng():
                torch.manual_seed(509)
                expected = _reference(kind, reference_model(inputs))
                expected.backward()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            for parameter, reference in zip(
                online.module.parameters(), reference_model.parameters(), strict=True
            ):
                torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-6, rtol=2e-5)
                assert parameter.grad.abs().sum() > 0

            # Neither shape mismatches nor a single bad rank may strand a peer.
            with pytest.raises(ValueError, match="equal local shapes"):
                regularizer(torch.ones(2, 3 + rank, 6), seed=11)
            invalid = torch.ones(2, 3, 6)
            if rank == 1:
                invalid[0, 0, 0] = float("nan")
            with pytest.raises(ValueError, match="invalid input on at least"):
                regularizer(invalid, seed=11)

            # Actual global batch of two supports one example on each rank.
            singleton = torch.ones(2, 1, 6, requires_grad=True)
            penalty = regularizer(singleton, seed=11)
            penalty.backward()
            assert torch.isfinite(singleton.grad).all()

            # Eval can run on only one rank without entering any collective.
            if rank == 0:
                regularizer.eval()
                assert torch.isfinite(regularizer(torch.ones(2, 3, 6), seed=11))
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo backend")
def test_two_process_gloo_matches_independent_global_batch_values_and_ddp_gradients(tmp_path):
    mp.spawn(
        _distributed_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("kind", ["sigreg", "localereg", "localereg_fixed_shape"])
def test_bfloat16_autocast_uses_fp32_statistics_and_preserves_gradients(kind):
    torch.manual_seed(83)
    inputs = torch.randn(2, 12, 4, requires_grad=True)
    producer = nn.Linear(4, 6)
    regularizer = _regularizer(kind)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        states = producer(inputs)
        assert states.dtype == torch.bfloat16
        actual = regularizer(states, seed=701)
    assert actual.dtype == torch.float32
    reference_states = states.detach().float().requires_grad_()
    with torch.random.fork_rng():
        torch.manual_seed(701)
        expected = _reference(kind, reference_states)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert inputs.grad.abs().sum() > 0
    assert torch.isfinite(inputs.grad).all()
    assert producer.weight.grad.abs().sum() > 0
    assert torch.isfinite(producer.weight.grad).all()


@pytest.mark.parametrize("kind", ["sigreg", "localereg", "localereg_fixed_shape"])
def test_explicit_seed_is_repeatable_without_consuming_global_rng(kind):
    regularizer = _regularizer(kind)
    states = torch.randn(2, 12, 6)
    before = torch.get_rng_state().clone()
    first = regularizer(states, seed=719)
    assert torch.equal(before, torch.get_rng_state())
    second = regularizer(states, seed=719)
    torch.testing.assert_close(first, second, atol=0, rtol=0)

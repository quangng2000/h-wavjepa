"""Evaluation keeps a fixed population when global batch 16 uses four/eight ranks."""

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_hierarchical_dynamics import model_factory, waveform

from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule
from src.objectives.localereg import LocaLeReg
from src.objectives.sigreg import SIGReg
from src.objectives.temporal_diagnostics import recording_variance


def _population_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world,
        init_method=rendezvous,
        timeout=timedelta(seconds=90),
    )
    try:
        torch.manual_seed(73)
        states = torch.randn(3, 16, 6)
        local = states[:, rank * (16 // world) : (rank + 1) * (16 // world)]
        # Lightning evaluates in inference mode. No training graph is necessary.
        with torch.inference_mode():
            for regularizer in (
                SIGReg(num_proj=8),
                LocaLeReg(
                    num_proj=8,
                    num_clusters=4,
                    min_cluster_size=4,
                    n_quad=32,
                    fixed_shape_kmeans=True,
                ),
            ):
                regularizer.eval()
                actual = regularizer(local, global_batch=True, seed=509)
                expected = regularizer(states, global_batch=False, seed=509)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert not actual.requires_grad
            values = local.transpose(0, 1)
            result = {
                "levels": [values],
                "dense_levels": [values],
                "predictions": [values[:, 0]],
                "targets": [values[:, 1]],
                "teacher_origins": [values[:, 2]],
            }
            gathered = HierarchicalWavJEPAModule.evaluation_population(None, result)
            torch.testing.assert_close(gathered["levels"][0], states.transpose(0, 1))
            torch.testing.assert_close(gathered["targets"][0], states[1])
            actual = recording_variance(gathered["dense_levels"][0])
            expected = recording_variance(states.transpose(0, 1))
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        # Default inference still works on rank zero alone without hanging peers.
        if rank == 0:
            assert torch.isfinite(SIGReg(num_proj=8).eval()(states, seed=509))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo")
def test_global_eval_regularizer_and_diagnostics_preserve_population_16(tmp_path, world):
    mp.spawn(
        _population_worker,
        args=(world, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=world,
        join=True,
    )


def test_synchronized_evaluation_is_explicit_and_never_changes_training():
    model = model_factory()
    with pytest.raises(ValueError, match="eval-mode"):
        model.compute_loss(waveform(2), synchronized_evaluation=True)
    model.eval()
    with (
        torch.inference_mode(),
        patch.object(model.regularizer, "forward", wraps=model.regularizer.forward) as regularize,
    ):
        model.compute_loss(waveform(2), seed=7)
        assert all(call.kwargs["global_batch"] is False for call in regularize.call_args_list)
        regularize.reset_mock()
        model.compute_loss(waveform(2), seed=7, synchronized_evaluation=True)
        assert all(call.kwargs["global_batch"] is True for call in regularize.call_args_list)


def test_lightning_validation_explicitly_selects_global_evaluation():
    model = model_factory().eval()
    audio = waveform(2)
    with torch.inference_mode():
        result = model.compute_loss(audio, seed=0, diagnostics=True)
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch.object(model, "compute_loss", return_value=result) as loss,
            patch.object(model, "evaluation_population", return_value=result) as population,
            patch.object(model, "log") as log,
        ):
            model.validation_step({"waveform": audio}, 0)
            assert loss.call_args.kwargs["synchronized_evaluation"] is True
            population.assert_called_once_with(result)
            assert any(call.args[0] == "val/diagnostic_recordings" for call in log.call_args_list)

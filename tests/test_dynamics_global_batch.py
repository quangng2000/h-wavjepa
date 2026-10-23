"""Controlled full-objective DDP parity, beyond merely equal states across ranks.

Masks are repeated per trajectory and dropout is zero so a split batch and a
single global batch see the same experiment. This does not assert identical
uncontrolled rank RNG/data-loader streams, nor BF16 kernel equivalence.
"""

from copy import deepcopy
from datetime import timedelta
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_hierarchical_dynamics import model_factory, waveform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.utils.runtime import require_finite_gradients


class _RepeatedTrajectoryMasker(nn.Module):
    def __init__(self, masker, windows):
        super().__init__()
        self.masker, self.windows = masker, windows

    def forward(self, batch, patches, channels, generator=None):
        assert batch % self.windows == 0
        masks = self.masker(self.windows, patches, channels, generator=np.random.default_rng(971))
        return tuple(mask.repeat(batch // self.windows, *([1] * (mask.ndim - 1))) for mask in masks)


class _LossForward(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, seed):
        return self.model.compute_loss(audio, seed=seed)["loss"]


def _global_batch_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=45)
    )
    try:
        for kind, hierarchy in (("sigreg", True), ("localereg", True), ("localereg", False)):
            torch.manual_seed(887)
            backbone_config = model_factory().backbone.construction_config()
            backbone_config.update(encoder_layers=2, average_top_k_layers=2)
            model = model_factory(
                backbone=WavJEPABackbone(config=backbone_config),
                regularizer_kind=kind,
                localereg_clusters=2,
                localereg_min_cluster_size=2,
                localereg_n_quad=32,
                hierarchy=hierarchy,
                readout_mode="normalized_top_k",
                optimizer=partial(torch.optim.SGD, lr=0.03, momentum=0.9),
            ).train()
            assert not any(
                isinstance(layer, nn.modules.batchnorm._BatchNorm) for layer in model.modules()
            )
            assert all(layer.p == 0 for layer in model.modules() if isinstance(layer, nn.Dropout))
            model.backbone.masker = _RepeatedTrajectoryMasker(
                model.backbone.masker, model.sample_windows
            )
            reference = deepcopy(model)
            ddp = DistributedDataParallel(_LossForward(model), find_unused_parameters=False)
            optimizer = model.configure_optimizers()
            reference_optimizer = reference.configure_optimizers()
            # Two iterations are needed to catch DDP's incomplete-reduction error
            # from any trainable parameter unused by the default active branches.
            for step in range(2):
                audio = waveform(4)
                seed = 983 + step
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
                actual = ddp(audio[rank * 2 : (rank + 1) * 2], seed)
                actual.backward()
                mean_actual = actual.detach().clone()
                dist.all_reduce(mean_actual)
                mean_actual /= 2
                # The reference independently assembles one global-batch objective.
                # Explicit per-level sketch seeds match the distributed estimator;
                # the normal standalone seeded path uses a different RNG sequence.
                reference.regularization_weight = 0
                with patch("torch.distributed.is_initialized", return_value=False):
                    result = reference.compute_loss(audio, seed=seed)
                    selected_levels = result["levels"] if hierarchy else result["levels"][:1]
                    penalty = sum(
                        reference.regularizer(
                            level.transpose(0, 1), global_batch=False, seed=seed + i
                        )
                        for i, level in enumerate(selected_levels)
                    )
                    expected = result["loss"] + model.regularization_weight * penalty
                    expected.backward()
                torch.testing.assert_close(mean_actual, expected, atol=2e-5, rtol=2e-5)
                actual_parameters = dict(model.named_parameters())
                for name, parameter in reference.named_parameters():
                    actual_parameter = actual_parameters[name]
                    if parameter.requires_grad:
                        assert parameter.grad is not None, name
                        assert actual_parameter.grad is not None, name
                        torch.testing.assert_close(
                            actual_parameter.grad,
                            parameter.grad,
                            atol=3e-5,
                            rtol=3e-4,
                            msg=lambda message, name=name: f"{kind}/{hierarchy}/{name}: {message}",
                        )
                require_finite_gradients(optimizer)
                optimizer.step()
                reference_optimizer.step()
                assert model.ema_updates == reference.ema_updates == step + 1
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(
                        value,
                        reference.state_dict()[name],
                        atol=1e-5,
                        rtol=1e-4,
                        msg=lambda message, name=name: f"{kind}/{hierarchy}/{name}: {message}",
                    )
            del ddp

        # A nonfinite input or loss on just one rank must reject on both ranks
        # before the next training collective or backward. No peer hangs.
        audio = waveform(2)
        if rank == 1:
            audio[0, 0] = float("nan")
        with pytest.raises(ValueError, match="at least one training rank"):
            model.compute_loss(audio, seed=11)
        clean_audio = waveform(2)
        original = model.backbone.masked_loss_with_targets

        def invalid_loss(*args, **kwargs):
            loss, targets = original(*args, **kwargs)
            return (loss * float("nan") if rank == 1 else loss), targets

        with patch.object(model.backbone, "masked_loss_with_targets", side_effect=invalid_loss):
            with pytest.raises(FloatingPointError, match="non-finite.*loss"):
                model.compute_loss(clean_audio, seed=11)
        parameter = nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        if rank == 1:
            parameter.grad = torch.full_like(parameter, float("inf"))
        # Rank zero deliberately has no gradient; it still joins the agreement.
        with pytest.raises(FloatingPointError, match="optimizer and EMA update aborted"):
            require_finite_gradients(optimizer)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo backend")
def test_two_rank_full_objective_gradients_updates_and_ema_match_one_global_batch(tmp_path):
    mp.spawn(_global_batch_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)

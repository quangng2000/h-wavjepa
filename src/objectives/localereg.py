"""LocaLeReg for audio: independent neighborhoods across recordings at each time.

The numerical loss is the pinned upstream implementation. Time is not treated
as a set of augmented views or flattened into the population. See docs/LOCALEREG.md.
"""

from functools import partial

import torch
from torch import nn

from src.objectives.distributed import prepare_population, regularizer_rng
from src.vendor.localereg.losses import ballreg_loss, local_ballreg_loss

SOURCE_COMMIT = "771361c18d37e65df3dc188761ffb26751ce4e37"


class LocaLeReg(nn.Module):
    def __init__(
        self,
        num_proj: int = 128,
        num_clusters: int = 4,
        min_cluster_size: int = 4,
        kmeans_iters: int = 10,
        n_quad: int = 256,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("num_proj", num_proj, 1),
            ("num_clusters", num_clusters, 1),
            ("min_cluster_size", min_cluster_size, 2),
            ("kmeans_iters", kmeans_iters, 1),
            ("n_quad", n_quad, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.num_proj = num_proj
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.kmeans_iters = kmeans_iters
        self.n_quad = n_quad
        self.last_stats: dict[str, float] = {}
        self._distributed_rng = torch.Generator().manual_seed(torch.initial_seed())

    def forward(
        self, states: torch.Tensor, *, global_batch: bool | None = None, seed: int | None = None
    ) -> torch.Tensor:
        if global_batch is None:
            global_batch = self.training and torch.is_grad_enabled()
        states, distributed = prepare_population(
            states, global_batch=global_batch, seed=seed, name="LocaLeReg"
        )
        with (
            torch.autocast(device_type=states.device.type, enabled=False),
            regularizer_rng(
                states, distributed=distributed, seed=seed, generator=self._distributed_rng
            ),
        ):
            return self._loss(states)

    def _loss(self, states: torch.Tensor) -> torch.Tensor:
        test = partial(ballreg_loss, center=True, num_slices=self.num_proj, n_quad=self.n_quad)
        losses, diagnostics = [], []
        for time_states in states:
            stats: dict[str, float] = {}
            losses.append(
                local_ballreg_loss(
                    time_states.unsqueeze(0),
                    loss=test,
                    num_clusters=self.num_clusters,
                    min_cluster_size=self.min_cluster_size,
                    kmeans_iters=self.kmeans_iters,
                    normalize_for_kmeans=True,
                    weighted=False,
                    kmeans_backend="torch",
                    stats=stats,
                )
            )
            diagnostics.append(stats)
        self.last_stats = {
            name: sum(stats[name] for stats in diagnostics) / len(diagnostics)
            for name in diagnostics[0]
        }
        return torch.stack(losses).mean()

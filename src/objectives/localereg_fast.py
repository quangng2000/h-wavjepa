"""Fixed-shape adaptation of LocaLeJEPA's detached torch k-means.

Adapted from src/losses.py at 771361c18d37e65df3dc188761ffb26751ce4e37.
Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo,
Randall Balestriero, Claudio Battiloro. MIT; see third_party/licenses/LocaLeJEPA.txt.

The byte-exact reference remains in src/vendor/localereg/losses.py. Fixed-size
scatter counts and where avoid dynamic boolean indexing inside Lloyd iterations.
Cluster filtering, equal weighting, RNG order and the supplied official BallReg
loss are unchanged. Dynamic cluster extraction still happens once per cluster.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.vendor.localereg.losses import _validate_kmeans_input


@torch.no_grad()
def fixed_shape_kmeans_labels(
    x: torch.Tensor, num_clusters: int, *, iters: int = 10, normalize: bool = True
) -> torch.Tensor:
    count, width, clusters = _validate_kmeans_input(x, num_clusters)
    x = x.float()
    if normalize:
        x = F.normalize(x, dim=-1)
    centroids = x[torch.randperm(count, device=x.device)[:clusters]]
    ones = torch.ones(count, dtype=torch.long, device=x.device)
    for _ in range(max(int(iters), 1)):
        labels = torch.cdist(x, centroids).argmin(dim=-1)
        sums = torch.zeros_like(centroids)
        sums.scatter_add_(0, labels[:, None].expand(-1, width), x)
        sizes = torch.zeros(clusters, device=x.device, dtype=torch.long)
        sizes.scatter_add_(0, labels, ones)
        centroids = torch.where((sizes > 0)[:, None], sums / sizes.clamp_min(1)[:, None], centroids)
        if normalize:
            centroids = F.normalize(centroids, dim=-1)
    return labels


def fixed_shape_local_loss(
    proj: torch.Tensor,
    *,
    loss: Callable[[torch.Tensor], torch.Tensor],
    num_clusters: int,
    min_cluster_size: int,
    kmeans_iters: int,
    stats: dict[str, float],
) -> torch.Tensor:
    """The audio adapter's exact upstream settings, with fixed-shape clustering.

    These settings are normalize_for_kmeans=True, weighted=False, num_global=None
    on a [views, recordings, width] population. Only assignment is detached.
    """
    if proj.ndim != 3 or proj.shape[1] < 1:
        raise ValueError("local_loss expects [views, nonempty recordings, width]")
    recordings = proj.shape[1]
    labels = fixed_shape_kmeans_labels(
        proj.mean(dim=0).detach(), num_clusters, iters=kmeans_iters, normalize=True
    )
    clusters = min(max(int(num_clusters), 1), recordings)
    sizes = torch.bincount(labels, minlength=clusters).tolist()
    losses, weights = [], []
    for cluster in range(clusters):
        if sizes[cluster] < max(int(min_cluster_size), 1):
            continue
        losses.append(loss(proj[:, labels == cluster, :]))
        weights.append(sizes[cluster])
    stats.update(
        configured_clusters=float(num_clusters),
        effective_clusters=float(len(losses)),
        kept_fraction=float(sum(weights)) / recordings,
        fallback=0.0 if losses else 1.0,
    )
    return torch.stack(losses).mean() if losses else loss(proj)

"""Local non-overlapping hierarchy with explicitly auditable sample supports."""

from collections.abc import Sequence
from math import prod
from numbers import Integral

import torch
from torch import Tensor, nn


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def patchify(waveform: Tensor, patch_size: int) -> Tensor:
    """Return [batch, patches, samples] without padding or temporal mixing."""
    patch_size = _positive_integer(patch_size, "patch_size")
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [B,T] or [B,1,T]")
    if not waveform.is_floating_point():
        raise ValueError("waveform must be floating point")
    if waveform.shape[1] == 0 or waveform.shape[1] % patch_size:
        raise ValueError("nonempty waveform length must be divisible by patch_size")
    if waveform.shape[0] == 0:
        raise ValueError("waveform batch must be nonempty")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform samples must be finite (no NaN or infinity)")
    return waveform.reshape(waveform.shape[0], -1, patch_size)


class HierarchyEncoder(nn.Module):
    """Patch-local features followed by local pooling, never full-context attention.

    Each higher token sees exactly ``prod(pool_factors[:level])`` raw patches.
    The same class with a single width and no pooling is the flat control.
    """

    def __init__(
        self,
        patch_size: int = 32,
        widths: Sequence[int] = (32, 48, 64),
        pool_factors: Sequence[int] = (2, 2),
    ) -> None:
        super().__init__()
        self.patch_size = _positive_integer(patch_size, "patch_size")
        self.widths = tuple(_positive_integer(width, "widths") for width in widths)
        self.pool_factors = tuple(
            _positive_integer(factor, "pool_factors") for factor in pool_factors
        )
        if not self.widths:
            raise ValueError("widths must contain at least one level")
        if len(self.pool_factors) != len(self.widths) - 1:
            raise ValueError("provide one pool factor for each upper level")
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.patch_size, self.widths[0]),
            nn.GELU(),
            nn.LayerNorm(self.widths[0]),
        )
        self.projections = nn.ModuleList(
            nn.Sequential(nn.Linear(low, high), nn.GELU(), nn.LayerNorm(high))
            for low, high in zip(self.widths[:-1], self.widths[1:], strict=True)
        )

    @property
    def total_pool_factor(self) -> int:
        return prod(self.pool_factors)

    def construction_config(self) -> dict[str, int | list[int]]:
        """Plain checkpoint metadata, including pooling semantics absent from weights."""
        return {
            "patch_size": self.patch_size,
            "widths": list(self.widths),
            "pool_factors": list(self.pool_factors),
        }

    def forward_patches(self, patches: Tensor) -> list[Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != self.patch_size:
            raise ValueError("patches must have shape [B,N,patch_size]")
        if patches.shape[0] == 0 or not patches.is_floating_point():
            raise ValueError("patches must be a nonempty floating point batch")
        if not torch.isfinite(patches).all():
            raise ValueError("patch samples must be finite (no NaN or infinity)")
        if patches.shape[1] == 0 or patches.shape[1] % self.total_pool_factor:
            raise ValueError("patch count must be divisible by the total pool factor")
        values = self.patch_encoder(patches)
        levels = [values]
        for factor, projection in zip(self.pool_factors, self.projections, strict=True):
            batch, count, width = values.shape
            values = values.reshape(batch, count // factor, factor, width).mean(dim=2)
            values = projection(values)
            levels.append(values)
        return levels

    def forward(self, waveform: Tensor) -> list[Tensor]:
        return self.forward_patches(patchify(waveform, self.patch_size))

    def propagate_mask(self, patch_mask: Tensor) -> list[Tensor]:
        """Any hidden raw support makes a coarse token unavailable as context."""
        if patch_mask.ndim != 2 or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be a boolean [B,N] tensor")
        if patch_mask.shape[1] == 0 or patch_mask.shape[1] % self.total_pool_factor:
            raise ValueError("mask length must be divisible by the total pool factor")
        masks = [patch_mask]
        mask = patch_mask
        for factor in self.pool_factors:
            mask = mask.reshape(mask.shape[0], -1, factor).any(dim=-1)
            masks.append(mask)
        return masks

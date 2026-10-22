"""Shared raw-support masking for hierarchical and flat comparisons."""

import torch
from torch import Tensor


def grouped_patch_mask(
    batch_size: int,
    patch_count: int,
    group_size: int,
    ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mask full aligned groups; every sample retains context and a target.

    Rounding means the realized ratio can differ from the requested ratio.
    The optional generator must match the device, following PyTorch's contract.
    """
    if group_size < 1 or patch_count % group_size:
        raise ValueError("patch_count must be divisible by positive group_size")
    if batch_size < 1 or not 0 < ratio < 1:
        raise ValueError("batch_size must be positive and mask ratio must be in (0,1)")
    group_count = patch_count // group_size
    if group_count < 2:
        raise ValueError("masking needs at least two groups: visible context and hidden target")
    hidden = min(group_count - 1, max(1, round(group_count * ratio)))
    ranks = torch.rand(batch_size, group_count, device=device, generator=generator).argsort(dim=1)
    groups = torch.zeros(batch_size, group_count, dtype=torch.bool, device=device)
    groups.scatter_(1, ranks[:, :hidden], True)
    return groups.repeat_interleave(group_size, dim=1)

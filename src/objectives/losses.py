"""Losses do not imply semantic preservation or optimizer convergence."""

from collections.abc import Sequence

import torch
from torch import Tensor


def masked_prediction_loss(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
    masks: Sequence[Tensor],
    weights: Sequence[float],
) -> tuple[Tensor, list[Tensor]]:
    """Mean over hidden tokens/features at each level, then weighted sum."""
    if not predictions or not len(predictions) == len(targets) == len(masks) == len(weights):
        raise ValueError("predictions, targets, masks, and weights must have matching levels")
    terms = []
    for prediction, target, mask in zip(predictions, targets, masks, strict=True):
        if prediction.shape != target.shape or mask.shape != prediction.shape[:2]:
            raise ValueError("prediction, target, and mask shapes must agree")
        if mask.dtype != torch.bool or not mask.any():
            raise ValueError("each prediction level needs a nonempty boolean target mask")
        terms.append((prediction - target.detach()).square().mean(dim=-1)[mask].mean())
    return sum(weight * term for weight, term in zip(weights, terms, strict=True)), terms


def reference_reconstruction_loss(reconstruction: Tensor, raw_patches: Tensor) -> Tensor:
    """The reference is the raw patch itself: no learned/EMA reference branch."""
    if reconstruction.shape != raw_patches.shape:
        raise ValueError("reference reconstruction must match raw patch shape")
    return (reconstruction - raw_patches.detach()).square().mean()


def feature_std(features: Tensor) -> Tensor:
    """Variation across examples/tokens, not the misleading within-token std."""
    return features.detach().float().flatten(0, 1).std(dim=0, unbiased=False).mean()

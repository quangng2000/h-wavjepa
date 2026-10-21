"""Lightning orchestration for a deliberately small H-WavJEPA prototype.

The reference reconstruction branch sees full waveforms but is isolated from
masked prediction. The Lean results do not certify this nonlinear optimizer,
its EMA targets, or downstream performance.
"""

import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from src.models.components.hierarchy import HierarchyEncoder, patchify
from src.models.components.masking import grouped_patch_mask
from src.models.components.predictor import MaskedPredictor
from src.objectives.losses import (
    feature_std,
    masked_prediction_loss,
    reference_reconstruction_loss,
)


class HWavJEPAModule(L.LightningModule):
    def __init__(
        self,
        encoder: HierarchyEncoder,
        optimizer: Callable[..., torch.optim.Optimizer],
        mask_ratio: float = 0.5,
        mask_group_size: int = 4,
        ema_decay: float = 0.99,
        prediction_weights: Sequence[float] | None = None,
        protection_weight: float = 1.0,
        protected_dim: int = 32,
    ) -> None:
        super().__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0,1)")
        if mask_group_size < 1 or mask_group_size % encoder.total_pool_factor:
            raise ValueError("mask_group_size must be a multiple of the total pool factor")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if not math.isfinite(protection_weight) or protection_weight < 0 or protected_dim < 1:
            raise ValueError("protection_weight must be finite/nonnegative; protected_dim positive")
        self.prediction_weights = tuple(
            float(weight)
            for weight in (
                prediction_weights
                if prediction_weights is not None
                else [1.0] * len(encoder.widths)
            )
        )
        if len(self.prediction_weights) != len(encoder.widths):
            raise ValueError("provide one prediction weight per encoder level")
        if any(not math.isfinite(w) or w < 0 for w in self.prediction_weights):
            raise ValueError("prediction weights must be finite and nonnegative")
        if not any(self.prediction_weights):
            raise ValueError("at least one prediction weight must be positive")
        # Keep checkpoint metadata plain: Hydra ListConfig objects otherwise
        # require unsafe pickle loading under PyTorch's weights-only default.
        self.save_hyperparameters(
            {
                "mask_ratio": float(mask_ratio),
                "mask_group_size": int(mask_group_size),
                "ema_decay": float(ema_decay),
                "prediction_weights": list(self.prediction_weights),
                "protection_weight": float(protection_weight),
                "protected_dim": int(protected_dim),
            }
        )
        self.encoder = encoder
        self.teacher = deepcopy(encoder).requires_grad_(False).eval()
        self.predictors = nn.ModuleList(MaskedPredictor(width) for width in encoder.widths)
        self.optimizer_factory = optimizer
        self.mask_ratio = mask_ratio
        self.mask_group_size = mask_group_size
        self.ema_decay = ema_decay
        self.protection_weight = protection_weight
        self.protected_encoder: nn.Module | None = None
        self.protected_decoder: nn.Module | None = None
        if protection_weight > 0:
            self.protected_encoder = nn.Sequential(
                nn.Linear(encoder.patch_size, protected_dim), nn.GELU()
            )
            self.protected_decoder = nn.Linear(protected_dim, encoder.patch_size)
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

    def train(self, mode: bool = True) -> "HWavJEPAModule":
        super().train(mode)
        self.teacher.eval()
        return self

    def make_patch_mask(self, waveform: Tensor) -> Tensor:
        patches = patchify(waveform, self.encoder.patch_size)
        return grouped_patch_mask(
            patches.shape[0],
            patches.shape[1],
            self.mask_group_size,
            self.mask_ratio,
            patches.device,
        )

    def predictive_outputs(self, waveform: Tensor, patch_mask: Tensor) -> dict[str, Any]:
        patches = patchify(waveform, self.encoder.patch_size)
        if patch_mask.shape != patches.shape[:2] or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be boolean [batch, raw patch count]")
        if patch_mask.device != patches.device:
            raise ValueError("patch_mask and waveform must be on the same device")
        masks = self.encoder.propagate_mask(patch_mask)
        for mask in masks:
            if not (mask.any(dim=1).all() and (~mask).any(dim=1).all()):
                raise ValueError("each sample needs visible and hidden tokens at every level")
        # Remove raw target information BEFORE any learned transformation.
        visible_patches = patches.masked_fill(patch_mask.unsqueeze(-1), 0.0)
        context = self.encoder.forward_patches(visible_patches)
        predictions = [
            predictor(level, mask)
            for predictor, level, mask in zip(self.predictors, context, masks, strict=True)
        ]
        with torch.no_grad():
            targets = self.teacher.forward_patches(patches)
        return {
            "predictions": predictions,
            "targets": targets,
            "masks": masks,
            "patch_mask": patch_mask,
        }

    def compute_loss(self, waveform: Tensor, patch_mask: Tensor | None = None) -> dict[str, Any]:
        if patch_mask is None:
            patch_mask = self.make_patch_mask(waveform)
        outputs = self.predictive_outputs(waveform, patch_mask)
        prediction, terms = masked_prediction_loss(
            outputs["predictions"], outputs["targets"], outputs["masks"], self.prediction_weights
        )
        protection = prediction.new_zeros(())
        if self.protected_encoder is not None and self.protected_decoder is not None:
            patches = patchify(waveform, self.encoder.patch_size)
            protected = self.protected_encoder(patches)
            protection = reference_reconstruction_loss(self.protected_decoder(protected), patches)
        return {
            **outputs,
            "loss": prediction + self.protection_weight * protection,
            "prediction_loss": prediction,
            "protection_loss": protection,
            "prediction_terms": terms,
        }

    @torch.no_grad()
    def encode(self, waveform: Tensor) -> dict[str, Any]:
        """Unmasked online features for readouts; protected tokens are NOT pooled.

        This is not a fixed-width concatenation: callers must explicitly choose
        alignment/pooling for their probe and account for its added dimensions.
        """
        patches = patchify(waveform, self.encoder.patch_size)
        levels = self.encoder.forward_patches(patches)
        protected = self.protected_encoder(patches) if self.protected_encoder is not None else None
        return {
            "levels": levels,
            "protected": protected,
            "readouts": [{"predictive": level, "protected": protected} for level in levels],
        }

    def forward(self, waveform: Tensor) -> dict[str, Any]:
        return self.encode(waveform)

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        waveform = batch["waveform"]
        result = self.compute_loss(waveform)
        log_kwargs = {
            "on_step": stage == "train",
            "on_epoch": True,
            "batch_size": waveform.shape[0],
            "sync_dist": True,
        }
        for key in ("loss", "prediction_loss", "protection_loss"):
            self.log(f"{stage}/{key}", result[key], prog_bar=key == "loss", **log_kwargs)
        for index, (target, prediction, loss) in enumerate(
            zip(result["targets"], result["predictions"], result["prediction_terms"], strict=True)
        ):
            self.log(f"{stage}/level_{index}/prediction_loss", loss, **log_kwargs)
            self.log(f"{stage}/level_{index}/target_std", feature_std(target), **log_kwargs)
            self.log(f"{stage}/level_{index}/prediction_std", feature_std(prediction), **log_kwargs)
        return result["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    @torch.no_grad()
    def update_teacher(self) -> None:
        for teacher, online in zip(
            self.teacher.parameters(), self.encoder.parameters(), strict=True
        ):
            teacher.lerp_(online, 1.0 - self.ema_decay)
        # No running-stat buffers currently exist; keep this contract safe if added.
        for teacher, online in zip(self.teacher.buffers(), self.encoder.buffers(), strict=True):
            teacher.copy_(online)
        self.ema_updates.add_(1)
        self.teacher.eval()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer_factory(params=(p for p in self.parameters() if p.requires_grad))
        # A post-step hook runs only when the underlying optimizer actually steps,
        # not on accumulation microbatches or AMP GradScaler-skipped updates.
        optimizer.register_step_post_hook(lambda _optimizer, _args, _kwargs: self.update_teacher())
        return optimizer

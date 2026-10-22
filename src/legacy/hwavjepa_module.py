"""Archived Lightning orchestration for the independent waveform prototype.

The reference reconstruction branch sees full waveforms but is isolated from
masked prediction. The Lean results do not certify this nonlinear optimizer,
its EMA targets, or downstream performance.
"""

import json
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn

from src.legacy.components.hierarchy import HierarchyEncoder, patchify
from src.legacy.components.masking import grouped_patch_mask
from src.legacy.components.predictor import MaskedPredictor
from src.legacy.losses import (
    feature_std,
    masked_prediction_loss,
    reference_reconstruction_loss,
)


class HWavJEPAModule(L.LightningModule):
    CHECKPOINT_KEY = "hwavjepa"
    CHECKPOINT_SCHEMA_VERSION = 1
    # Bump the schema when these implementations change incompatibly. Tensor
    # shapes alone cannot identify pooling, masking, or objective semantics.
    CHECKPOINT_SEMANTICS = {
        "encoder": "local_mean_pool_v1",
        "predictor": "masked_transformer_v1",
        "objective": "masked_mse_raw_patch_protection_v1",
        "evaluation_mask": "cpu_generator_seed_plus_batch_index_v1",
    }

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
        eval_mask_seed: int = 0,
        input_sample_rate: int | None = None,
    ) -> None:
        super().__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0,1)")
        if (
            type(mask_group_size) is not int
            or mask_group_size < 1
            or mask_group_size % encoder.total_pool_factor
        ):
            raise ValueError("mask_group_size must be a multiple of the total pool factor")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if (
            not math.isfinite(protection_weight)
            or protection_weight < 0
            or type(protected_dim) is not int
            or protected_dim < 1
        ):
            raise ValueError("protection_weight must be finite/nonnegative; protected_dim positive")
        if type(eval_mask_seed) is not int or not 0 <= eval_mask_seed < 2**63:
            raise ValueError("eval_mask_seed must be an integer in [0, 2**63)")
        if input_sample_rate is not None and (
            type(input_sample_rate) is not int or input_sample_rate <= 0
        ):
            raise ValueError("input_sample_rate must be a positive integer or None")
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
                "eval_mask_seed": eval_mask_seed,
                "input_sample_rate": input_sample_rate,
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
        self.eval_mask_seed = eval_mask_seed
        self.input_sample_rate = input_sample_rate
        self.protected_encoder: nn.Module | None = None
        self.protected_decoder: nn.Module | None = None
        if protection_weight > 0:
            self.protected_encoder = nn.Sequential(
                nn.Linear(encoder.patch_size, protected_dim), nn.GELU()
            )
            self.protected_decoder = nn.Linear(protected_dim, encoder.patch_size)
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Serializable construction/semantic contract, without callable objects.

        Optimizer and data-run contracts are separate from this model contract.
        Construct an inference model from ``encoder`` and ``module`` only after
        validating a trusted checkpoint with ``validated_checkpoint_metadata``.
        """
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "semantics": dict(self.CHECKPOINT_SEMANTICS),
            "encoder": self.encoder.construction_config(),
            "module": dict(self.hparams),
        }

    @classmethod
    def validated_checkpoint_metadata(cls, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Validate the versioned model metadata before constructing/loading weights.

        This validates the format, not the trustworthiness of a checkpoint file.
        Old unversioned checkpoints require an explicit audited migration; their
        pooling/objective configuration cannot be inferred safely from tensors.
        """
        metadata = checkpoint.get(cls.CHECKPOINT_KEY)
        if not isinstance(metadata, dict):
            raise ValueError(
                "checkpoint is missing H-WavJEPA model metadata; unversioned checkpoints "
                "require an explicit audited migration with their original configuration"
            )
        if (
            type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported H-WavJEPA checkpoint schema_version")
        if set(metadata) != {"schema_version", "semantics", "encoder", "module"}:
            raise ValueError("invalid H-WavJEPA checkpoint metadata fields")
        if metadata["semantics"] != cls.CHECKPOINT_SEMANTICS:
            raise ValueError("incompatible H-WavJEPA checkpoint implementation semantics")
        fields = {
            "encoder": {"patch_size", "widths", "pool_factors"},
            "module": {
                "mask_ratio",
                "mask_group_size",
                "ema_decay",
                "prediction_weights",
                "protection_weight",
                "protected_dim",
                "eval_mask_seed",
                "input_sample_rate",
            },
        }
        for section, names in fields.items():
            if not isinstance(metadata[section], dict) or set(metadata[section]) != names:
                raise ValueError(f"invalid H-WavJEPA checkpoint {section} fields")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint model metadata must contain finite plain data") from error
        return deepcopy(metadata)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[self.CHECKPOINT_KEY] = self.checkpoint_metadata()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        saved = self.validated_checkpoint_metadata(checkpoint)
        current = self.checkpoint_metadata()
        mismatches = [
            f"{section}.{name}"
            for section in ("encoder", "module")
            for name, value in current[section].items()
            if saved[section][name] != value
        ]
        if mismatches:
            raise ValueError(
                "checkpoint model configuration mismatch: "
                + ", ".join(mismatches)
                + "; restore the original configuration instead of silently changing semantics"
            )

    def train(self, mode: bool = True) -> "HWavJEPAModule":
        super().train(mode)
        self.teacher.eval()
        return self

    def make_patch_mask(self, waveform: Tensor, *, seed: int | None = None) -> Tensor:
        """Draw training masks, or repeatable evaluation masks using a local RNG.

        A supplied seed uses a CPU generator and moves the resulting boolean
        mask to the input device. It never consumes global CPU/accelerator RNG.
        Evaluation defaults to ``eval_mask_seed``; Lightning adds ``batch_idx``.
        Masks are intentionally batch-layout-sensitive: keep batch size/order
        and distributed sharding fixed when comparing evaluation measurements.
        """
        patches = patchify(waveform, self.encoder.patch_size)
        if seed is None and not self.training:
            seed = self.eval_mask_seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("mask seed must be an integer in [0, 2**63)")
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        mask = grouped_patch_mask(
            patches.shape[0],
            patches.shape[1],
            self.mask_group_size,
            self.mask_ratio,
            patches.device if generator is None else torch.device("cpu"),
            generator=generator,
        )
        return mask.to(device=patches.device)

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
        loss = prediction + self.protection_weight * protection
        if not torch.isfinite(torch.stack([loss, prediction, protection, *terms])).all():
            raise FloatingPointError(
                "non-finite H-WavJEPA loss; check waveform scale, model weights, "
                "learning rate, and precision before continuing training"
            )
        return {
            **outputs,
            "loss": loss,
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

    def _shared_step(self, batch: dict[str, Tensor], stage: str, batch_idx: int = 0) -> Tensor:
        waveform = batch["waveform"]
        mask = None
        if stage != "train":
            seed = (self.eval_mask_seed + batch_idx) % 2**63
            mask = self.make_patch_mask(waveform, seed=seed)
        result = self.compute_loss(waveform, mask)
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
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Reject bad unscaled gradients before optimizer/EMA state can change.

        A finite loss does not imply finite derivatives. Lightning invokes this
        hook after accumulation and before the optimizer step. When a precision
        plugin supplies a GradScaler, leave nonfinite detection and skipped-step
        recovery to that scaler. The EMA post-step hook separately checks the
        fused optimizer's ``found_inf`` flag before updating the teacher.
        """
        precision = getattr(self._trainer, "precision_plugin", None)
        if getattr(precision, "scaler", None) is not None:
            return
        optimized = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        for name, parameter in self.named_parameters():
            if id(parameter) not in optimized or parameter.grad is None:
                continue
            gradient = parameter.grad
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not torch.isfinite(values).all():
                raise FloatingPointError(
                    f"non-finite gradient in parameter '{name}'; optimizer and EMA update "
                    "aborted before changing state. Check waveform scale, learning rate, "
                    "and precision."
                )

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
        optimizer.register_step_post_hook(self._after_optimizer_step)
        return optimizer

    def _after_optimizer_step(
        self, optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict
    ) -> None:
        """Advance EMA only after a real update, including fused AMP optimizers.

        Ordinary GradScaler skips do not invoke ``optimizer.step``. Fused
        optimizers do invoke it but skip the update inside their kernel: their
        transient ``found_inf`` flag remains available while post-hooks run.
        """
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None and torch.as_tensor(found_inf).ne(0).any():
            return
        self.update_teacher()

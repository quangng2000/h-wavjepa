"""Hierarchical multi-horizon continuation of the official WavJEPA core.

WavJEPA's masked objective remains intact. Additional heads forecast future
windows in distinct learned spaces, and their gradients reach the backbone.
This observational audio extension is not H-JEPA's action-conditioned planner.

References and adaptation choices: docs/UPSTREAM_DYNAMICS.md. Upper state
encoders follow H-JEPA's experimental window_size=1 (pointwise), not temporal
pooling; its per-level strides [2, 2] are cumulative state_strides [1, 2, 4] here.
"""

import json
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import get_cosine_schedule_with_warmup

from src.models.components.temporal_predictor import TemporalPredictor
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.objectives.sigreg import SOURCE_COMMIT as SIGREG_COMMIT
from src.objectives.sigreg import SIGReg


class HierarchicalWavJEPAModule(L.LightningModule):
    """One complete trajectory per example, with a shared past-only origin.

    For context_windows=C, target index at level l is C-1+horizons[l].
    State stride subsamples past states, not raw samples or target windows. The
    same final observed window must be retained at every level. G2/G3 are pointwise
    MLPs; their stride is not a temporal pooling window. All levels have the same
    width. Flat control uses identity G2/G3 but identical horizons and strides.
    """

    CHECKPOINT_KEY = "hierarchical_wavjepa"
    CHECKPOINT_SCHEMA_VERSION = 1
    SEMANTICS = "upstream_masked_plus_past_only_window_forecast_sigreg_v1"
    SCHEDULE_CHECKPOINT_KEY = "hierarchical_wavjepa_training_schedule"
    # Missing fields in early schema-v1 checkpoints retain their old behavior.
    OPTIONAL_CONFIG_DEFAULTS = {
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "lr_total_steps": None,
        "ema_end_decay": None,
        "ema_anneal_end_step": None,
    }

    def __init__(
        self,
        backbone: WavJEPABackbone,
        optimizer: Callable[..., torch.optim.Optimizer],
        context_windows: int = 5,
        horizons: Sequence[int] = (1, 2, 4),
        state_strides: Sequence[int] = (1, 2, 4),
        hierarchy: bool = True,
        hidden_dim: int = 256,
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        prediction_weights: Sequence[float] = (1.0, 1.0, 1.0),
        masked_weight: float = 1.0,
        regularization_weight: float = 0.1,
        ema_decay: float = 0.999,
        eval_seed: int = 0,
        sigreg_projections: int = 128,
        lr_schedule: str = "constant",
        lr_warmup_steps: int | None = 0,
        lr_total_steps: int | None = None,
        ema_end_decay: float | None = None,
        ema_anneal_end_step: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("context_windows", context_windows),
            ("hidden_dim", hidden_dim),
            ("predictor_layers", predictor_layers),
            ("predictor_heads", predictor_heads),
            ("sigreg_projections", sigreg_projections),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if context_windows < 2:
            raise ValueError("context_windows must be >= 2")
        horizons, state_strides = tuple(horizons), tuple(state_strides)
        if len(horizons) != 3 or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError("horizons must contain three positive integers")
        if not horizons[0] < horizons[1] < horizons[2]:
            raise ValueError("horizons must be strictly increasing")
        if len(state_strides) != 3 or any(
            type(s) is not int or s < 1 or (context_windows - 1) % s for s in state_strides
        ):
            raise ValueError("state_strides must align every level to the final context window")
        if state_strides[0] != 1 or not state_strides[0] <= state_strides[1] <= state_strides[2]:
            raise ValueError("state_strides must start at 1 and be nondecreasing")
        if any(high % low for low, high in zip(state_strides[:-1], state_strides[1:])):
            raise ValueError("state_strides must be nested integer multiples of the prior level")
        if type(hierarchy) is not bool:
            raise ValueError("hierarchy must be boolean")
        if type(eval_seed) is not int or not 0 <= eval_seed < 2**32:
            raise ValueError("eval_seed must be an integer in [0, 2**32)")
        prediction_weights = tuple(float(w) for w in prediction_weights)
        if len(prediction_weights) != 3:
            raise ValueError("prediction_weights must have three entries")
        if any(
            not math.isfinite(w) or w < 0
            for w in (*prediction_weights, masked_weight, regularization_weight)
        ):
            raise ValueError("loss weights must be finite and nonnegative")
        if not any(prediction_weights) and masked_weight <= 0:
            raise ValueError("at least one prediction objective must be active")
        if hierarchy and any(prediction_weights[1:]) and regularization_weight <= 0:
            raise ValueError("learned upper-level targets require positive regularization_weight")
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be constant or cosine")
        for name, value, minimum in (
            ("lr_warmup_steps", lr_warmup_steps, 0),
            ("lr_total_steps", lr_total_steps, 1),
            ("ema_anneal_end_step", ema_anneal_end_step, 1),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum} or null")
        if ema_end_decay is not None and (
            not math.isfinite(ema_end_decay) or not ema_decay <= ema_end_decay < 1
        ):
            raise ValueError("ema_end_decay must be finite and in [ema_decay,1)")
        if ema_anneal_end_step is not None and ema_end_decay is None:
            raise ValueError("ema_anneal_end_step requires ema_end_decay")
        self.backbone = backbone
        self.context_windows = context_windows
        self.horizons = horizons
        self.state_strides = state_strides
        self.window_samples = backbone.window_samples
        self.sample_windows = context_windows + max(horizons)
        self.required_samples = self.sample_windows * self.window_samples
        self.prediction_weights = prediction_weights
        self.masked_weight = float(masked_weight)
        self.regularization_weight = float(regularization_weight)
        self.ema_decay = float(ema_decay)
        self.eval_seed = eval_seed
        self.optimizer_factory = optimizer
        self._resolved_training_schedule: dict[str, int] | None = None
        width = backbone.embedding_dim
        self.abstractions = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, width))
            if hierarchy
            else nn.Identity()
            for _ in range(2)
        )
        self.teacher_abstractions = deepcopy(self.abstractions).requires_grad_(False).eval()
        self.predictors = nn.ModuleList(
            TemporalPredictor(width, context_windows, predictor_layers, predictor_heads)
            if weight > 0
            else nn.Identity()
            for weight in prediction_weights
        )
        self.regularizer = SIGReg(num_proj=sigreg_projections)
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        self.config = {
            "context_windows": context_windows,
            "horizons": list(horizons),
            "state_strides": list(state_strides),
            "hierarchy": hierarchy,
            "hidden_dim": hidden_dim,
            "predictor_layers": predictor_layers,
            "predictor_heads": predictor_heads,
            "prediction_weights": list(prediction_weights),
            "masked_weight": self.masked_weight,
            "regularization_weight": self.regularization_weight,
            "ema_decay": self.ema_decay,
            "eval_seed": eval_seed,
            "sigreg_projections": sigreg_projections,
            "lr_schedule": lr_schedule,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_total_steps": lr_total_steps,
            "ema_end_decay": ema_end_decay,
            "ema_anneal_end_step": ema_anneal_end_step,
        }
        self.save_hyperparameters(deepcopy(self.config))

    def setup(self, stage: str) -> None:
        if self.trainer.world_size != 1 or str(self.trainer.precision) not in {"32", "32-true"}:
            raise ValueError("Upstream dynamics integration currently supports single-device FP32")
        if self.trainer.strategy.root_device.type not in {"cpu", "cuda"}:
            raise ValueError("Upstream dynamics integration currently supports CPU/CUDA only")

    def train(self, mode: bool = True) -> "HierarchicalWavJEPAModule":
        super().train(mode)
        self.teacher_abstractions.eval()
        self.backbone.model.teacher_encoder.eval()
        return self

    def on_train_start(self) -> None:
        if self.config["lr_schedule"] == "cosine":
            total = self._resolve_training_schedule()["total_steps"]
            if self.trainer.estimated_stepping_batches > total:
                raise ValueError(
                    "Trainer exceeds the saved cosine schedule budget; choose the full "
                    "lr_total_steps before training, and resume within that budget"
                )

    def _windows(self, waveform: Tensor, expected: int | None = None) -> Tensor:
        if waveform.ndim != 2 or waveform.shape[0] < 1:
            raise ValueError("waveform must be nonempty [batch, samples]")
        if waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain finite float32 samples")
        if not waveform.shape[1] or waveform.shape[1] % self.window_samples:
            raise ValueError("waveform must contain complete contiguous audio windows")
        windows = waveform.reshape(waveform.shape[0], -1, self.window_samples)
        if expected is not None and windows.shape[1] != expected:
            raise ValueError(f"expected exactly {expected} windows, got {windows.shape[1]}")
        return windows

    @staticmethod
    def _levels(
        low: Tensor,
        abstractions: nn.ModuleList,
        state_strides: Sequence[int] = (1, 1, 1),
    ) -> list[Tensor]:
        """Compose pointwise encoders on successively subsampled state streams.

        No neighboring states are averaged. With window_size=1, subsampling
        before each MLP is equivalent to selecting its dense outputs afterward.
        """
        result = [low]
        for abstract, previous, current in zip(
            abstractions, state_strides[:-1], state_strides[1:], strict=True
        ):
            result.append(abstract(result[-1][:, :: current // previous]))
        return result

    def encode(self, waveform: Tensor, *, subsample: bool = False) -> list[Tensor]:
        """Differentiable contextual window embeddings, not generated audio.

        Each completed window is encoded independently. Features inside a window
        become available only at its end, despite the backbone's finer token grid.
        Use eval() and torch.no_grad() explicitly for downstream feature extraction.

        The default retains dense embeddings for the existing inference API.
        subsample=True returns each level on its training-time temporal grid:
        level l has floor((window_count - 1) / state_strides[l]) + 1 states.
        """
        return self._levels(
            self.backbone.encode_windows(self._windows(waveform)),
            self.abstractions,
            self.state_strides if subsample else (1, 1, 1),
        )

    @torch.no_grad()
    def encode_targets(self, waveform: Tensor, *, subsample: bool = False) -> list[Tensor]:
        """EMA states on a dense or successively subsampled temporal grid."""
        return self._levels(
            self.backbone.encode_teacher_windows(self._windows(waveform)),
            self.teacher_abstractions,
            self.state_strides if subsample else (1, 1, 1),
        )

    def _predict_levels(self, levels: list[Tensor]) -> list[Tensor]:
        """Consume already-subsampled levels with their original window positions."""
        predictions = []
        for level, stride, predictor, weight in zip(
            levels, self.state_strides, self.predictors, self.prediction_weights, strict=True
        ):
            positions = torch.arange(0, self.context_windows, stride, device=level.device)
            predictions.append(predictor(level, positions) if weight > 0 else level[:, -1] * 0)
        return predictions

    def predict(self, waveform: Tensor) -> list[Tensor]:
        self._windows(waveform, self.context_windows)
        return self._predict_levels(self.encode(waveform, subsample=True))

    def forward(self, waveform: Tensor) -> list[Tensor]:
        return self.predict(waveform)

    def compute_loss(self, waveform: Tensor, seed: int | None = None) -> dict[str, Any]:
        windows = self._windows(waveform, self.sample_windows)
        if seed is None and not self.training:
            seed = self.eval_seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0,2**32)")
        zero = waveform.new_zeros(())
        masked = self.backbone.masked_loss(windows, seed=seed) if self.masked_weight else zero
        terms, reg_terms = [zero] * 3, [zero] * 3
        levels, predictions, targets = [], [], []
        if any(self.prediction_weights) or self.regularization_weight:
            # Encode ONLY the prefix for the online forecasting path. The separate
            # masked anchor may use all windows, but shares no contextual activations.
            prefix = waveform[:, : self.context_windows * self.window_samples]
            levels = self.encode(prefix, subsample=True)
            predictions = self._predict_levels(levels)
            target_levels = self.encode_targets(waveform)
            targets = [
                values[:, self.context_windows - 1 + horizon]
                for values, horizon in zip(target_levels, self.horizons, strict=True)
            ]
            terms = [
                F.mse_loss(prediction, target) if weight > 0 else zero
                for prediction, target, weight in zip(
                    predictions, targets, self.prediction_weights, strict=True
                )
            ]
            if self.regularization_weight:
                # fork_rng makes evaluation repeatable without consuming global
                # RNG. CPU evaluation is the reference; accelerator RNG is seeded
                # separately below. MPS evaluation is rejected until RNG isolation
                # can be guaranteed by this helper.
                if seed is not None and waveform.device.type not in {"cpu", "cuda"}:
                    raise ValueError("Seeded SIGReg evaluation currently supports CPU/CUDA only")
                devices = [waveform.device.index or 0] if waveform.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices, enabled=seed is not None):
                    if seed is not None:
                        torch.random.default_generator.manual_seed(seed)
                        for device in devices:
                            torch.cuda.default_generators[device].manual_seed(seed)
                    reg_terms = [self.regularizer(level.transpose(0, 1)) for level in levels]
        prediction_loss = sum(
            (weight * loss for weight, loss in zip(self.prediction_weights, terms, strict=True)),
            start=zero,
        )
        regularization_loss = sum(reg_terms, start=zero)
        loss = (
            self.masked_weight * masked
            + prediction_loss
            + self.regularization_weight * regularization_loss
        )
        if not torch.isfinite(torch.stack([loss, masked, *terms, *reg_terms])).all():
            raise FloatingPointError("non-finite upstream WavJEPA dynamics loss")
        return {
            "loss": loss,
            "masked_loss": masked,
            "prediction_loss": prediction_loss,
            "regularization_loss": regularization_loss,
            "prediction_terms": terms,
            "regularization_terms": reg_terms,
            "levels": levels,
            "predictions": predictions,
            "targets": targets,
        }

    def _shared_step(self, batch: dict[str, Tensor], stage: str, batch_idx: int) -> Tensor:
        seed = None if stage == "train" else (self.eval_seed + batch_idx) % 2**32
        result = self.compute_loss(batch["waveform"], seed=seed)
        kwargs = dict(on_step=stage == "train", on_epoch=True, batch_size=len(batch["waveform"]))
        for key in ("loss", "masked_loss", "prediction_loss", "regularization_loss"):
            self.log(f"{stage}/{key}", result[key], prog_bar=key == "loss", **kwargs)
        for i, (term, penalty) in enumerate(
            zip(result["prediction_terms"], result["regularization_terms"], strict=True)
        ):
            self.log(f"{stage}/level_{i}/prediction_loss", term, **kwargs)
            self.log(f"{stage}/level_{i}/sigreg", penalty, **kwargs)
        for i, level in enumerate(result["levels"]):
            self.log(
                f"{stage}/level_{i}/batch_std", level.std(dim=0, unbiased=False).mean(), **kwargs
            )
        return result["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(
                        "non-finite gradient; optimizer and EMA update aborted"
                    )

    @torch.no_grad()
    def update_teacher(self) -> None:
        decay = self.current_ema_decay()
        self.backbone.update_masked_teacher(decay)
        for teacher, online in zip(
            self.teacher_abstractions.parameters(), self.abstractions.parameters(), strict=True
        ):
            teacher.lerp_(online, 1 - decay)
        self.teacher_abstractions.eval()
        self.ema_updates.add_(1)

    def current_ema_decay(self) -> float:
        """Official WavJEPA linear annealing, indexed by completed optimizer steps.

        Adapted from the pinned JEPA._get_ema_decay implementation. The outer
        Lightning module owns the step counter because the vendored core is a
        child module. Accumulated batches and failed steps do not advance it.
        """
        end_decay = self.config["ema_end_decay"]
        if end_decay is None:
            return self.ema_decay
        end_step = self.config["ema_anneal_end_step"]
        if self._resolved_training_schedule is not None:
            end_step = self._resolved_training_schedule["ema_end_step"]
        if end_step is None:
            raise RuntimeError("EMA annealing needs configure_optimizers or ema_anneal_end_step")
        step = int(self.ema_updates.item())
        if step >= end_step:
            return end_decay
        remaining = 1 - step / end_step
        return end_decay - (end_decay - self.ema_decay) * remaining

    def _after_optimizer_step(
        self, optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict
    ) -> None:
        found_inf = getattr(optimizer, "found_inf", None)
        if found_inf is not None and torch.as_tensor(found_inf).ne(0).any():
            return
        self.update_teacher()

    def _resolve_training_schedule(self) -> dict[str, int]:
        if self._resolved_training_schedule is not None:
            return self._resolved_training_schedule
        total = self.config["lr_total_steps"]
        if total is None:
            if self._trainer is None:
                raise RuntimeError("automatic training schedule needs an attached Trainer")
            total = self.trainer.estimated_stepping_batches
        if not math.isfinite(total) or int(total) != total or total < 1:
            raise ValueError("training schedule requires a finite positive optimizer-step budget")
        total = int(total)
        warmup = self.config["lr_warmup_steps"]
        # The official implementation fixes warmup at 100k. Our configurable
        # adaptation uses 10% by default so short experiments can finish warmup.
        warmup = int(0.1 * total) if warmup is None else warmup
        if warmup > total:
            raise ValueError("lr_warmup_steps cannot exceed the total optimizer-step budget")
        schedule = {
            "total_steps": total,
            "warmup_steps": warmup,
            "ema_end_step": self.config["ema_anneal_end_step"] or total,
        }
        self._resolved_training_schedule = schedule
        return schedule

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        optimizer = self.optimizer_factory(params=(p for p in self.parameters() if p.requires_grad))
        optimizer.register_step_post_hook(self._after_optimizer_step)
        if self.config["lr_schedule"] == "cosine" or self.config["ema_end_decay"] is not None:
            schedule = self._resolve_training_schedule()
        if self.config["lr_schedule"] == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=schedule["warmup_steps"],
                num_training_steps=schedule["total_steps"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.CHECKPOINT_SCHEMA_VERSION,
            "semantics": self.SEMANTICS,
            "upstream_commit": self.backbone.source_commit,
            "sigreg_commit": SIGREG_COMMIT,
            "backbone": self.backbone.construction_config(),
            "dynamics": deepcopy(self.config),
        }

    @classmethod
    def validated_checkpoint_metadata(cls, checkpoint: dict) -> dict:
        metadata = checkpoint.get(cls.CHECKPOINT_KEY)
        if not isinstance(metadata, dict) or set(metadata) != {
            "schema_version",
            "semantics",
            "upstream_commit",
            "sigreg_commit",
            "backbone",
            "dynamics",
        }:
            raise ValueError("missing or invalid hierarchical WavJEPA checkpoint metadata")
        if (
            type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
            or metadata["semantics"] != cls.SEMANTICS
        ):
            raise ValueError("incompatible hierarchical WavJEPA checkpoint semantics")
        if not isinstance(metadata["backbone"], dict) or not isinstance(metadata["dynamics"], dict):
            raise ValueError("checkpoint construction metadata must be plain dictionaries")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint metadata must contain finite plain data") from error
        normalized = deepcopy(metadata)
        normalized["dynamics"] = cls.OPTIONAL_CONFIG_DEFAULTS | normalized["dynamics"]
        return normalized

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint[self.CHECKPOINT_KEY] = self.checkpoint_metadata()
        if self._resolved_training_schedule is not None:
            checkpoint[self.SCHEDULE_CHECKPOINT_KEY] = deepcopy(self._resolved_training_schedule)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.validated_checkpoint_metadata(checkpoint) != self.checkpoint_metadata():
            raise ValueError(
                "checkpoint model configuration mismatch; use the original dynamics configuration"
            )
        schedule = checkpoint.get(self.SCHEDULE_CHECKPOINT_KEY)
        if schedule is not None:
            if (
                not isinstance(schedule, dict)
                or set(schedule) != {"total_steps", "warmup_steps", "ema_end_step"}
                or any(type(value) is not int for value in schedule.values())
                or schedule["total_steps"] < 1
                or schedule["ema_end_step"] < 1
                or not 0 <= schedule["warmup_steps"] <= schedule["total_steps"]
            ):
                raise ValueError("invalid checkpoint training schedule")
            for field, key in (
                ("lr_total_steps", "total_steps"),
                ("lr_warmup_steps", "warmup_steps"),
                ("ema_anneal_end_step", "ema_end_step"),
            ):
                configured = self.config[field]
                if configured is not None and configured != schedule[key]:
                    raise ValueError(f"checkpoint training schedule disagrees with {field}")
            # Keep the original budget when resuming with a different Trainer
            # stopping point. Lightning restores the optimizer/LambdaLR state.
            self._resolved_training_schedule = deepcopy(schedule)

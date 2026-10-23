"""Hierarchical multi-horizon continuation of the official WavJEPA core.

WavJEPA's masked objective remains intact. Additional heads forecast future
windows in distinct learned spaces, and their gradients reach the backbone.
This observational audio extension is not H-JEPA's action-conditioned planner.

References and adaptation choices: docs/UPSTREAM_DYNAMICS.md. The pointwise
baseline follows H-JEPA's experimental window_size=1. The optional temporal
version pools causal, valid windows of lower states before each upper map.
"""

import json
import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from torch import Tensor, nn
from transformers import get_cosine_schedule_with_warmup

from src.models.components.temporal_predictor import TemporalPredictor
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.objectives.localereg import SOURCE_COMMIT as LOCALEREG_COMMIT
from src.objectives.localereg import LocaLeReg
from src.objectives.sigreg import SOURCE_COMMIT as SIGREG_COMMIT
from src.objectives.sigreg import SIGReg
from src.objectives.temporal_diagnostics import forecast_diagnostics, recording_variance
from src.objectives.temporal_variance import temporal_variance_floor
from src.utils.runtime import all_ranks_true, require_finite_gradients


class HierarchicalWavJEPAModule(L.LightningModule):
    """One complete trajectory per example, with a shared past-only origin.

    For context_windows=C, target index at level l is C-1+horizons[l].
    State stride subsamples past states, not raw samples or target windows. The
    same final observed window must be retained at every level. Each upper map
    optionally pools a causal window of lower states on that lower level's grid.
    Only states with complete support exist; no boundary padding is introduced.
    All levels have the same width. Pointwise behavior is the default baseline.
    """

    CHECKPOINT_KEY = "hierarchical_wavjepa"
    CHECKPOINT_SCHEMA_VERSION = 1
    SEMANTICS = "upstream_masked_plus_past_only_window_forecast_sigreg_v1"
    LOCALEREG_SEMANTICS = "upstream_masked_plus_past_only_window_forecast_localereg_v1"
    SCHEDULE_CHECKPOINT_KEY = "hierarchical_wavjepa_training_schedule"
    # Missing fields in early schema-v1 checkpoints retain their old behavior.
    OPTIONAL_CONFIG_DEFAULTS = {
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "lr_total_steps": None,
        "ema_end_decay": None,
        "ema_anneal_end_step": None,
        "regularizer_kind": "sigreg",
        "localereg_clusters": 4,
        "localereg_min_cluster_size": 4,
        "localereg_kmeans_iters": 10,
        "localereg_n_quad": 256,
        "localereg_fixed_shape_kmeans": False,
        "readout_mode": "legacy",
        "regularization_scope": "per_level",
        "aggregation_windows": [1, 1, 1],
        "temporal_variance_weight": 0.0,
        "temporal_std_floor": 0.1,
        "temporal_variance_epsilon": 1e-4,
        "masked_crops_per_recording": 0,
        "masked_crop_samples": None,
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
        regularizer_kind: str = "sigreg",
        localereg_clusters: int = 4,
        localereg_min_cluster_size: int = 4,
        localereg_kmeans_iters: int = 10,
        localereg_n_quad: int = 256,
        localereg_fixed_shape_kmeans: bool = False,
        readout_mode: str = "legacy",
        training_metric_interval: int = 1,
        regularization_scope: str = "unique_spaces",
        aggregation_windows: Sequence[int] = (1, 1, 1),
        temporal_variance_weight: float = 0.0,
        temporal_std_floor: float = 0.1,
        temporal_variance_epsilon: float = 1e-4,
        masked_crops_per_recording: int = 0,
        masked_crop_samples: int | None = None,
    ) -> None:
        super().__init__()
        if type(masked_crops_per_recording) is not int or masked_crops_per_recording < 0:
            raise ValueError("masked_crops_per_recording must be a nonnegative integer")
        if masked_crops_per_recording:
            if (
                type(masked_crop_samples) is not int
                or masked_crop_samples < 1
                or masked_weight <= 0
            ):
                raise ValueError("Separate masked crops require a positive length and masked loss")
        elif masked_crop_samples is not None:
            raise ValueError("masked_crop_samples requires masked_crops_per_recording")
        self.masked_crops_per_recording = masked_crops_per_recording
        self.masked_crop_samples = masked_crop_samples
        if type(training_metric_interval) is not int or training_metric_interval < 1:
            raise ValueError("training_metric_interval must be a positive integer")
        # This changes observation frequency, not the loss, optimizer or model
        # state. Keep it out of construction metadata so a checkpoint can resume
        # with a different logging cadence without changing its research setup.
        self.training_metric_interval = training_metric_interval
        self._initial_training_step = 0
        if type(localereg_fixed_shape_kmeans) is not bool:
            raise ValueError("localereg_fixed_shape_kmeans must be boolean")
        if readout_mode not in {"legacy", "normalized_top_k"}:
            raise ValueError("readout_mode must be legacy or normalized_top_k")
        self.readout_mode = readout_mode
        if regularization_scope not in {"unique_spaces", "per_level"}:
            raise ValueError("regularization_scope must be unique_spaces or per_level")
        self.regularization_scope = regularization_scope
        if regularizer_kind not in {"sigreg", "localereg"}:
            raise ValueError("regularizer_kind must be sigreg or localereg")
        for name, value, minimum in (
            ("localereg_clusters", localereg_clusters, 1),
            ("localereg_min_cluster_size", localereg_min_cluster_size, 2),
            ("localereg_kmeans_iters", localereg_kmeans_iters, 1),
            ("localereg_n_quad", localereg_n_quad, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
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
        aggregation_windows = tuple(aggregation_windows)
        if (
            len(aggregation_windows) != 3
            or any(type(window) is not int or window < 1 for window in aggregation_windows)
            or aggregation_windows[0] != 1
        ):
            raise ValueError(
                "aggregation_windows must contain three positive integers starting at 1"
            )
        support_windows = [1]
        for window, stride in zip(aggregation_windows[1:], state_strides[:-1], strict=True):
            support_windows.append(support_windows[-1] + (window - 1) * stride)
        if max(support_windows) > context_windows:
            raise ValueError("every aggregation level must have complete support in the context")
        if type(hierarchy) is not bool:
            raise ValueError("hierarchy must be boolean")
        for name, value in (
            ("temporal_variance_weight", temporal_variance_weight),
            ("temporal_std_floor", temporal_std_floor),
            ("temporal_variance_epsilon", temporal_variance_epsilon),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if type(eval_seed) is not int or not 0 <= eval_seed < 2**32:
            raise ValueError("eval_seed must be an integer in [0, 2**32)")
        prediction_weights = tuple(float(w) for w in prediction_weights)
        if len(prediction_weights) != 3:
            raise ValueError("prediction_weights must have three entries")
        if any(
            not math.isfinite(w) or w < 0
            for w in (
                *prediction_weights,
                masked_weight,
                regularization_weight,
                temporal_variance_weight,
            )
        ):
            raise ValueError("loss weights must be finite and nonnegative")
        if not any(prediction_weights) and masked_weight <= 0:
            raise ValueError("at least one prediction objective must be active")
        if hierarchy and any(prediction_weights[1:]) and regularization_weight <= 0:
            raise ValueError("learned upper-level targets require positive regularization_weight")
        if not math.isfinite(temporal_variance_epsilon) or temporal_variance_epsilon <= 0:
            raise ValueError("temporal_variance_epsilon must be finite and positive")
        if not math.isfinite(temporal_std_floor) or temporal_std_floor <= math.sqrt(
            temporal_variance_epsilon
        ):
            raise ValueError("temporal_std_floor must be finite and exceed sqrt(epsilon)")
        if temporal_variance_weight > 0 and (not hierarchy or not all(prediction_weights[1:])):
            raise ValueError("temporal variance requires learned upper-level prediction spaces")
        if temporal_variance_weight > 0 and any(
            context_windows - support + 1 < 2 for support in support_windows[1:]
        ):
            raise ValueError(
                "temporal variance requires at least two dense valid states per upper level"
            )
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
        self.aggregation_windows = aggregation_windows
        self._level_support_windows = tuple(support_windows)
        self.temporal_variance_weight = float(temporal_variance_weight)
        self.temporal_std_floor = float(temporal_std_floor)
        self.temporal_variance_epsilon = float(temporal_variance_epsilon)
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
        self.regularizer_kind = regularizer_kind
        self.regularizer = (
            LocaLeReg(
                num_proj=sigreg_projections,
                num_clusters=localereg_clusters,
                min_cluster_size=localereg_min_cluster_size,
                kmeans_iters=localereg_kmeans_iters,
                n_quad=localereg_n_quad,
                fixed_shape_kmeans=localereg_fixed_shape_kmeans,
            )
            if regularizer_kind == "localereg"
            else SIGReg(num_proj=sigreg_projections)
        )
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))
        self.config = {
            "context_windows": context_windows,
            "horizons": list(horizons),
            "state_strides": list(state_strides),
            "aggregation_windows": list(aggregation_windows),
            "temporal_variance_weight": self.temporal_variance_weight,
            "temporal_std_floor": self.temporal_std_floor,
            "temporal_variance_epsilon": self.temporal_variance_epsilon,
            "hierarchy": hierarchy,
            "hidden_dim": hidden_dim,
            "predictor_layers": predictor_layers,
            "predictor_heads": predictor_heads,
            "prediction_weights": list(prediction_weights),
            "masked_weight": self.masked_weight,
            "masked_crops_per_recording": self.masked_crops_per_recording,
            "masked_crop_samples": self.masked_crop_samples,
            "regularization_weight": self.regularization_weight,
            "regularization_scope": regularization_scope,
            "ema_decay": self.ema_decay,
            "eval_seed": eval_seed,
            "sigreg_projections": sigreg_projections,
            "regularizer_kind": regularizer_kind,
            "localereg_clusters": localereg_clusters,
            "localereg_min_cluster_size": localereg_min_cluster_size,
            "localereg_kmeans_iters": localereg_kmeans_iters,
            "localereg_n_quad": localereg_n_quad,
            "localereg_fixed_shape_kmeans": localereg_fixed_shape_kmeans,
            "readout_mode": readout_mode,
            "lr_schedule": lr_schedule,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_total_steps": lr_total_steps,
            "ema_end_decay": ema_end_decay,
            "ema_anneal_end_step": ema_anneal_end_step,
        }
        self.save_hyperparameters(deepcopy(self.config))
        self.save_hyperparameters({"training_metric_interval": training_metric_interval})

    def setup(self, stage: str) -> None:
        if str(self.trainer.precision) not in {"32", "32-true", "bf16-mixed"}:
            raise ValueError("Upstream dynamics supports FP32 or bf16-mixed precision")
        if not isinstance(self.trainer.strategy, (SingleDeviceStrategy, DDPStrategy)):
            raise ValueError("Upstream dynamics supports single-device or DDP execution")
        if self.trainer.strategy.root_device.type not in {"cpu", "cuda"}:
            raise ValueError("Upstream dynamics integration currently supports CPU/CUDA only")

    def train(self, mode: bool = True) -> "HierarchicalWavJEPAModule":
        super().train(mode)
        self.teacher_abstractions.eval()
        self.backbone.model.teacher_encoder.eval()
        return self

    def on_train_start(self) -> None:
        # Always observe the first optimizer step of a run, including resumes
        # that stop before the next periodic diagnostic interval.
        self._initial_training_step = getattr(self._trainer, "global_step", 0)
        if self.config["lr_schedule"] == "cosine":
            total = self._resolve_training_schedule()["total_steps"]
            if self.trainer.estimated_stepping_batches > total:
                raise ValueError(
                    "Trainer exceeds the saved cosine schedule budget; choose the full "
                    "lr_total_steps before training, and resume within that budget"
                )

    def _windows(self, waveform: Tensor, expected: int | None = None) -> Tensor:
        channels = self.backbone.config.get("input_channels", 1)
        if waveform.ndim != (2 if channels == 1 else 3) or waveform.shape[0] < 1:
            raise ValueError("waveform must be [batch, samples] or binaural [batch, 2, samples]")
        if channels == 2 and waveform.shape[1] != 2:
            raise ValueError("Binaural waveforms require exactly two channels")
        if waveform.dtype != torch.float32 or not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain finite float32 samples")
        if not waveform.shape[-1] or waveform.shape[-1] % self.window_samples:
            raise ValueError("waveform must contain complete contiguous audio windows")
        windows = waveform.unfold(-1, self.window_samples, self.window_samples)
        if channels == 2:
            windows = windows.transpose(1, 2)
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

    @property
    def level_support_windows(self) -> tuple[int, ...]:
        """Span of low audio windows needed by each completed upper state."""
        return self._level_support_windows

    def level_positions(
        self, window_count: int, *, subsample: bool = False, device: Any = None
    ) -> list[Tensor]:
        """Physical low-window endpoints for the valid states returned by encode."""
        if type(window_count) is not int or window_count < 0:
            raise ValueError("window_count must be a nonnegative integer")
        positions = []
        for support, stride in zip(self.level_support_windows, self.state_strides, strict=True):
            endpoints = torch.arange(min(support - 1, window_count), window_count, device=device)
            positions.append(
                endpoints[endpoints.remainder(stride) == 0] if subsample else endpoints
            )
        return positions

    def _subsample_levels(self, levels: list[Tensor], window_count: int) -> list[Tensor]:
        return [
            level[:, positions - support + 1]
            for level, positions, support in zip(
                levels,
                self.level_positions(window_count, subsample=True, device=levels[0].device),
                self.level_support_windows,
                strict=True,
            )
        ]

    def levels_from_low(
        self, low: Tensor, *, teacher: bool = False, subsample: bool = False
    ) -> list[Tensor]:
        """Compose valid causal upper states without padding or future input.

        Upper windows sample the previous dense stream at its cumulative stride.
        For strides [1,2,4] and aggregation [1,2,2], G2 sees low[t-1:t+1]
        and G3 sees medium[t-2] and medium[t], covering four low windows.
        Dense output includes every valid endpoint, even between training-grid
        endpoints; short inputs return empty upper states, not padded features.
        """
        if low.ndim != 3:
            raise ValueError("low states must have shape [batch, time, feature]")
        abstractions = self.teacher_abstractions if teacher else self.abstractions
        if self.aggregation_windows == (1, 1, 1):
            # Preserve the baseline's original operation ordering and numerics.
            return self._levels(low, abstractions, self.state_strides if subsample else (1, 1, 1))
        result = [low]
        for abstract, window, stride in zip(
            abstractions, self.aggregation_windows[1:], self.state_strides[:-1], strict=True
        ):
            previous = result[-1]
            offset = (window - 1) * stride
            count = max(0, previous.shape[1] - offset)
            if count == 0:
                pooled = previous[:, :0]
            elif window == 1:
                pooled = previous
            else:
                pooled = torch.stack(
                    [
                        previous[:, offset - j * stride : offset - j * stride + count]
                        for j in range(window)
                    ],
                    dim=0,
                ).mean(dim=0)
            result.append(abstract(pooled))
        return self._subsample_levels(result, low.shape[1]) if subsample else result

    def encode(self, waveform: Tensor, *, subsample: bool = False) -> list[Tensor]:
        """Differentiable contextual window embeddings, not generated audio.

        Each completed window is encoded independently. Features inside a window
        become available only at its end, despite the backbone's finer token grid.
        Use eval() and torch.no_grad() explicitly for downstream feature extraction.

        The default returns dense valid embeddings. With temporal aggregation,
        each level omits its incomplete prefix. subsample=True additionally keeps
        endpoints divisible by that level's cumulative state stride.
        """
        return self.levels_from_low(
            self.backbone.encode_windows(self._windows(waveform), readout_mode=self.readout_mode),
            subsample=subsample,
        )

    @torch.no_grad()
    def encode_targets(self, waveform: Tensor, *, subsample: bool = False) -> list[Tensor]:
        """EMA states on a dense or successively subsampled temporal grid."""
        return self.levels_from_low(
            self.backbone.encode_teacher_windows(self._windows(waveform)),
            teacher=True,
            subsample=subsample,
        )

    def _predict_levels(self, levels: list[Tensor]) -> list[Tensor]:
        """Consume already-subsampled levels with their original window positions."""
        predictions = []
        for level, positions, predictor, weight in zip(
            levels,
            self.level_positions(self.context_windows, subsample=True, device=levels[0].device),
            self.predictors,
            self.prediction_weights,
            strict=True,
        ):
            predictions.append(predictor(level, positions) if weight > 0 else level[:, -1] * 0)
        return predictions

    def predict(self, waveform: Tensor) -> list[Tensor]:
        self._windows(waveform, self.context_windows)
        return self._predict_levels(self.encode(waveform, subsample=True))

    def forward(self, waveform: Tensor) -> list[Tensor]:
        return self.predict(waveform)

    def compute_loss(
        self,
        waveform: Tensor,
        seed: int | None = None,
        *,
        diagnostics: bool = False,
        masked_crops: Tensor | None = None,
    ) -> dict[str, Any]:
        distributed_training = (
            self.training
            and torch.is_grad_enabled()
            and dist.is_available()
            and dist.is_initialized()
        )
        channels = self.backbone.config.get("input_channels", 1)
        forecasting = bool(
            any(self.prediction_weights)
            or self.regularization_weight
            or self.temporal_variance_weight
        )
        valid_shape = (
            waveform.ndim == (2 if channels == 1 else 3)
            and waveform.shape[0] > 0
            and waveform.shape[-1] > 0
            and (channels == 1 or waveform.shape[1] == 2)
            and (
                not forecasting
                and self.masked_crops_per_recording
                or waveform.shape[-1] == self.required_samples
            )
            and waveform.dtype == torch.float32
        )
        valid = torch.isfinite(waveform).all() & bool(valid_shape)
        if self.masked_crops_per_recording:
            expected = (waveform.shape[0], self.masked_crops_per_recording)
            expected += (() if channels == 1 else (2,)) + (self.masked_crop_samples,)
            valid_crops = isinstance(masked_crops, Tensor) and tuple(masked_crops.shape) == expected
            valid_crops = (
                valid_crops
                and masked_crops.dtype == torch.float32
                and masked_crops.device == waveform.device
            )
            valid &= torch.isfinite(masked_crops).all() if valid_crops else False
        elif masked_crops is not None:
            valid &= False
        if distributed_training:
            if not all_ranks_true(valid):
                raise ValueError("invalid waveform or masked crops on at least one training rank")
        elif not bool(valid):
            raise ValueError("Invalid waveform shape, finite float32 samples or masked crop recipe")
        windows = (
            self._windows(waveform, self.sample_windows)
            if forecasting or not self.masked_crops_per_recording
            else None
        )
        if seed is None and not self.training:
            seed = self.eval_seed
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0,2**32)")
        zero = waveform.new_zeros(())
        teacher_windows = None
        if self.masked_weight:
            if self.masked_crops_per_recording:
                masked = self.backbone.masked_loss(masked_crops, seed=seed)
            else:
                masked, teacher_windows = self.backbone.masked_loss_with_targets(windows, seed=seed)
        else:
            masked = zero
        terms, reg_terms, temporal_terms = [zero] * 3, [zero] * 3, [zero] * 3
        levels, dense_levels, predictions, targets = [], [], [], []
        regularization_stats = []
        teacher_origins = []
        if (
            any(self.prediction_weights)
            or self.regularization_weight
            or self.temporal_variance_weight
        ):
            # Encode ONLY the prefix for the online forecasting path. The separate
            # masked anchor may use all windows, but shares no contextual activations.
            prefix = waveform[..., : self.context_windows * self.window_samples]
            if self.aggregation_windows != (1, 1, 1) or self.temporal_variance_weight:
                dense_levels = self.encode(prefix)
                levels = self._subsample_levels(dense_levels, self.context_windows)
            else:
                # Retain baseline floating-point operation order exactly.
                levels = self.encode(prefix, subsample=True)
                # Evaluation variance compares the same dense prefix across
                # variants. Reuse low states; this diagnostic-only branch never
                # changes predictions, regularization, or the training graph.
                dense_levels = self.levels_from_low(levels[0]) if diagnostics else levels
            predictions = self._predict_levels(levels)
            if any(self.prediction_weights):
                indices = [self.context_windows - 1 + horizon for horizon in self.horizons]
                with torch.no_grad():
                    # The masked anchor already evaluated the dense teacher. Reuse
                    # only detached targets; no future activations enter the student.
                    # Aggregated targets need their full causal support. The
                    # no-anchor path encodes all windows once and shares them
                    # with origin diagnostics. The pointwise baseline keeps its
                    # original three-window optimization and numerical order.
                    if self.aggregation_windows != (1, 1, 1):
                        dense_teacher = (
                            teacher_windows
                            if teacher_windows is not None
                            else self.backbone.encode_teacher_windows(windows)
                        )
                        target_levels = self.levels_from_low(dense_teacher, teacher=True)
                        targets = [
                            values[:, endpoint - support + 1]
                            for values, endpoint, support in zip(
                                target_levels, indices, self.level_support_windows, strict=True
                            )
                        ]
                        if diagnostics:
                            teacher_origins = [
                                values[:, self.context_windows - support]
                                for values, support in zip(
                                    target_levels, self.level_support_windows, strict=True
                                )
                            ]
                    else:
                        selected = (
                            teacher_windows[:, indices]
                            if teacher_windows is not None
                            else self.backbone.encode_teacher_windows(windows[:, indices])
                        )
                        target_levels = self._levels(selected, self.teacher_abstractions)
                        targets = [values[:, i] for i, values in enumerate(target_levels)]
                        if diagnostics:
                            origin = self.context_windows - 1
                            origin_low = (
                                teacher_windows[:, origin : origin + 1]
                                if teacher_windows is not None
                                else self.backbone.encode_teacher_windows(
                                    windows[:, origin : origin + 1]
                                )
                            )
                            teacher_origins = [
                                values[:, 0]
                                for values in self._levels(origin_low, self.teacher_abstractions)
                            ]
                terms = [
                    F.mse_loss(prediction.float(), target.float()) if weight > 0 else zero
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
                    raise ValueError("Seeded regularization currently supports CPU/CUDA only")
                devices = [waveform.device.index or 0] if waveform.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices, enabled=seed is not None):
                    if seed is not None:
                        torch.random.default_generator.manual_seed(seed)
                        for device in devices:
                            torch.cuda.default_generators[device].manual_seed(seed)
                    reg_terms = []
                    distributed = dist.is_available() and dist.is_initialized()
                    for index, level in enumerate(levels):
                        # Flat levels are slices of the same low state space.
                        # Regularize its full context grid once, not overlapping
                        # stride grids three times. Hierarchical levels remain
                        # distinct distributions and each keeps its own penalty.
                        # Legacy checkpoints explicitly retain per_level behavior.
                        if (
                            self.regularization_scope == "unique_spaces"
                            and not self.config["hierarchy"]
                            and index > 0
                            and self.aggregation_windows[index] == 1
                        ):
                            reg_terms.append(zero)
                            regularization_stats.append({})
                            continue
                        # The same random statistic must be evaluated on every
                        # rank's differentiably gathered population. Accumulation
                        # does not enlarge that population. Single-device legacy
                        # runs retain their original seeded RNG sequence.
                        regularizer_seed = (
                            (self.eval_seed + 3 * self.global_step + index) % 2**32
                            if distributed and self.training and seed is None
                            else (seed + index) % 2**32
                            if distributed and seed is not None
                            else None
                        )
                        reg_terms.append(
                            self.regularizer(
                                level.transpose(0, 1),
                                global_batch=self.training and torch.is_grad_enabled(),
                                seed=regularizer_seed,
                            )
                        )
                        regularization_stats.append(
                            dict(getattr(self.regularizer, "last_stats", {}))
                        )
            if self.temporal_variance_weight:
                temporal_terms = [zero] + [
                    temporal_variance_floor(
                        level,
                        std_floor=self.temporal_std_floor,
                        epsilon=self.temporal_variance_epsilon,
                    )
                    for level in dense_levels[1:]
                ]
        prediction_loss = sum(
            (weight * loss for weight, loss in zip(self.prediction_weights, terms, strict=True)),
            start=zero,
        )
        regularization_loss = sum(reg_terms, start=zero)
        temporal_variance_loss = sum(temporal_terms, start=zero)
        loss = (
            self.masked_weight * masked
            + prediction_loss
            + self.regularization_weight * regularization_loss
        )
        if self.temporal_variance_weight:
            loss = loss + self.temporal_variance_weight * temporal_variance_loss
        if not all_ranks_true(
            torch.isfinite(torch.stack([loss, masked, *terms, *reg_terms, *temporal_terms])).all(),
            synchronize=distributed_training,
        ):
            raise FloatingPointError("non-finite upstream WavJEPA dynamics loss")
        return {
            "loss": loss,
            "masked_loss": masked,
            "prediction_loss": prediction_loss,
            "regularization_loss": regularization_loss,
            "temporal_variance_loss": temporal_variance_loss,
            "prediction_terms": terms,
            "regularization_terms": reg_terms,
            "temporal_variance_terms": temporal_terms,
            "regularization_stats": regularization_stats,
            "levels": levels,
            "dense_levels": dense_levels,
            "predictions": predictions,
            "targets": targets,
            "teacher_origins": teacher_origins,
        }

    @torch.no_grad()
    def temporal_diagnostics(self, result: dict[str, Any]) -> list[dict[str, Tensor]]:
        """Cheap held-out probes from already encoded states and EMA targets.

        No extra backbone pass is needed when the default masked anchor is active.
        Only small predictor heads see the cyclically shifted context. These
        measurements reveal temporal collapse; they do not prevent or certify it.
        """
        levels = result["levels"]
        stats = [recording_variance(level) for level in result.get("dense_levels", levels)]
        for stat, level, dense in zip(
            stats, levels, result.get("dense_levels", levels), strict=True
        ):
            stat["variance_context_states"] = level.new_tensor(dense.shape[1])
            stat["predictor_context_states"] = level.new_tensor(level.shape[1])
        if not result["teacher_origins"]:
            return stats
        shifted_predictions = self._predict_levels([level.roll(1, dims=1) for level in levels])
        for index, (prediction, target, origin, shifted, weight) in enumerate(
            zip(
                result["predictions"],
                result["targets"],
                result["teacher_origins"],
                shifted_predictions,
                self.prediction_weights,
                strict=True,
            )
        ):
            if weight > 0:
                stats[index].update(forecast_diagnostics(prediction, target, origin, shifted))
        return stats

    def _shared_step(self, batch: dict[str, Tensor], stage: str, batch_idx: int) -> Tensor:
        seed = None if stage == "train" else (self.eval_seed + batch_idx) % 2**32
        result = self.compute_loss(
            batch["waveform"],
            seed=seed,
            diagnostics=stage != "train",
            masked_crops=batch.get("masked_crops"),
        )
        if (
            stage == "train"
            and self.global_step != self._initial_training_step
            and (self.global_step + 1) % self.training_metric_interval
        ):
            # Lightning's log_every_n_steps only throttles logger output; it does
            # not suppress self.log(sync_dist=True) collectives. Sample training
            # diagnostics explicitly. Epoch training means then describe these
            # sampled batches; validation/test still include every batch.
            return result["loss"]
        kwargs = dict(
            on_step=stage == "train",
            on_epoch=True,
            batch_size=len(batch["waveform"]),
            sync_dist=True,
        )
        for key in (
            "loss",
            "masked_loss",
            "prediction_loss",
            "regularization_loss",
            "temporal_variance_loss",
        ):
            self.log(f"{stage}/{key}", result[key], prog_bar=key == "loss", **kwargs)
        for i, (term, penalty) in enumerate(
            zip(result["prediction_terms"], result["regularization_terms"], strict=True)
        ):
            self.log(f"{stage}/level_{i}/prediction_loss", term, **kwargs)
            self.log(f"{stage}/level_{i}/{self.regularizer_kind}", penalty, **kwargs)
            self.log(
                f"{stage}/level_{i}/temporal_variance_loss",
                result["temporal_variance_terms"][i],
                **kwargs,
            )
        for i, stats in enumerate(result["regularization_stats"]):
            for name, value in stats.items():
                self.log(f"{stage}/level_{i}/localereg_{name}", value, **kwargs)
        for i, level in enumerate(result["levels"]):
            self.log(
                f"{stage}/level_{i}/batch_std", level.std(dim=0, unbiased=False).mean(), **kwargs
            )
        if stage != "train":
            for i, stats in enumerate(self.temporal_diagnostics(result)):
                for name, value in stats.items():
                    self.log(f"{stage}/level_{i}/{name}", value, **kwargs)
        return result["loss"]

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        require_finite_gradients(optimizer)

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
            "semantics": (
                self.LOCALEREG_SEMANTICS if self.regularizer_kind == "localereg" else self.SEMANTICS
            ),
            "upstream_commit": self.backbone.source_commit,
            "sigreg_commit": SIGREG_COMMIT,
            **(
                {"localereg_commit": LOCALEREG_COMMIT}
                if self.regularizer_kind == "localereg"
                else {}
            ),
            "backbone": self.backbone.construction_config(),
            "dynamics": deepcopy(self.config),
        }

    @classmethod
    def validated_checkpoint_metadata(cls, checkpoint: dict) -> dict:
        metadata = checkpoint.get(cls.CHECKPOINT_KEY)
        if not isinstance(metadata, dict):
            raise ValueError("missing or invalid hierarchical WavJEPA checkpoint metadata")
        is_local = metadata.get("semantics") == cls.LOCALEREG_SEMANTICS
        required = {
            "schema_version",
            "semantics",
            "upstream_commit",
            "sigreg_commit",
            "backbone",
            "dynamics",
        } | ({"localereg_commit"} if is_local else set())
        if set(metadata) != required:
            raise ValueError("missing or invalid hierarchical WavJEPA checkpoint metadata")
        if (
            type(metadata["schema_version"]) is not int
            or metadata["schema_version"] != cls.CHECKPOINT_SCHEMA_VERSION
            or metadata["semantics"] not in {cls.SEMANTICS, cls.LOCALEREG_SEMANTICS}
        ):
            raise ValueError("incompatible hierarchical WavJEPA checkpoint semantics")
        if not isinstance(metadata["backbone"], dict) or not isinstance(metadata["dynamics"], dict):
            raise ValueError("checkpoint construction metadata must be plain dictionaries")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint metadata must contain finite plain data") from error
        normalized = deepcopy(metadata)
        normalized["backbone"].setdefault("compact_masked_tokens", False)
        normalized["backbone"].setdefault("input_channels", 1)
        normalized["dynamics"] = cls.OPTIONAL_CONFIG_DEFAULTS | normalized["dynamics"]
        if normalized["dynamics"]["regularizer_kind"] != (
            "localereg" if is_local else "sigreg"
        ) or (is_local and metadata["localereg_commit"] != LOCALEREG_COMMIT):
            raise ValueError("checkpoint regularizer kind or source revision mismatch")
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

"""Small run-safety helpers; Lightning continues to own the training loop."""

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def all_ranks_true(flag: torch.Tensor, *, synchronize: bool = True) -> bool:
    """Agree on a scalar validity flag before any rank takes a failure branch.

    Call only at a boundary reached by every training rank. Evaluation and
    arbitrary exceptions inside a rank's forward do not use this collective.
    """
    if synchronize and dist.is_available() and dist.is_initialized():
        flag = flag.detach().to(dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag)


def require_finite_gradients(optimizer: torch.optim.Optimizer) -> None:
    """Reject nonfinite gradients with one host synchronization per device.

    Testing each CUDA parameter in a Python conditional serializes the host with
    hundreds of kernels. Reduce the per-parameter flags on device first. Unlike
    using a gradient norm, this cannot reject large but finite gradients because
    the norm overflows. Sparse gradients are checked without densifying them.
    """
    flags = defaultdict(list)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is not None:
                values = gradient._values() if gradient.is_sparse else gradient
                flags[gradient.device].append(torch.isfinite(values).all())
    parameter = next((p for group in optimizer.param_groups for p in group["params"]), None)
    if parameter is None:
        return
    # One scalar collective also covers a rank with no gradient tensors. Every
    # rank then aborts before its optimizer/EMA hook, instead of stranding peers.
    finite = torch.ones((), dtype=torch.bool, device=parameter.device)
    for device_flags in flags.values():
        finite = finite & torch.stack(device_flags).all().to(parameter.device)
    if not all_ranks_true(finite):
        raise FloatingPointError("non-finite gradient; optimizer and EMA update aborted")


def atomic_json(path: Path, value: dict) -> None:
    """Replace a local JSON artifact only after its complete contents are flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    with Path(path).expanduser().open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def model_config_for_checkpoint(cfg: DictConfig) -> DictConfig:
    """A complete Lightning checkpoint owns all weights on resume/evaluation.

    Avoid downloading or requiring an initialization file that will immediately
    be replaced. Do not mutate the user's requested config or checkpoint schema.
    """
    if cfg.get("ckpt_path") and cfg.model.get("_target_") == (
        "src.models.hierarchical_wavjepa_module.HierarchicalWavJEPAModule"
    ):
        return OmegaConf.merge(
            cfg.model,
            {"backbone": {"checkpoint_path": None, "pretrained": False, "pretrained_path": None}},
        )
    return cfg.model


def validate_config(cfg: DictConfig) -> None:
    """Fail before model allocation for incompatible audio/masking settings."""
    optimizer = cfg.model.get("optimizer")
    if (
        isinstance(optimizer, DictConfig)
        and optimizer.get("_target_") in {"torch.optim.Adam", "torch.optim.AdamW"}
        and optimizer.get("fused", False)
        and str(cfg.trainer.get("precision", "32-true")).endswith("-mixed")
        and cfg.trainer.get("gradient_clip_val", 0)
    ):
        raise ValueError(
            "Lightning mixed precision rejects fused Adam/AdamW with gradient clipping; "
            "set model.optimizer.fused=false and retain the configured clipping"
        )
    if cfg.get("num_threads") is not None and (
        type(cfg.num_threads) is not int or cfg.num_threads < 1
    ):
        raise ValueError("num_threads must be a positive integer or null")
    if cfg.get("seed") is not None and (type(cfg.seed) is not int or not 0 <= cfg.seed < 2**32):
        raise ValueError("seed must be an integer in [0, 2**32) or null")
    strategy = cfg.trainer.get("strategy", "auto")
    parent_only_state = isinstance(strategy, str) and strategy.startswith(
        ("ddp_spawn", "ddp_fork", "ddp_notebook")
    )
    if isinstance(strategy, DictConfig):
        target = strategy.get("_target_", "")
        if isinstance(target, str) and target.rsplit(".", 1)[-1] == "DDPStrategy":
            parent_only_state = strategy.get("start_method", "popen") in {
                "spawn",
                "fork",
                "forkserver",
            }
    if parent_only_state:
        raise ValueError(
            "Spawn/fork/notebook DDP strategies are not supported by run artifact tracking: "
            "the parent Trainer can retain stale progress and checkpoint paths. "
            "Use single-device training or externally launched DDP with appropriate validation."
        )
    if (
        cfg.model.get("_target_")
        == "src.models.hierarchical_wavjepa_module.HierarchicalWavJEPAModule"
    ):
        _validate_upstream_forecasting(cfg)
        return
    # This is a H-WavJEPA input contract, not a restriction on arbitrary Hydra models.
    if cfg.model.get("_target_") != "src.models.hwavjepa_module.HWavJEPAModule":
        return
    samples, patch, group = (
        cfg.data.num_samples,
        cfg.model.encoder.patch_size,
        cfg.model.mask_group_size,
    )
    for name, value in (
        ("data.num_samples", samples),
        ("patch_size", patch),
        ("mask_group_size", group),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    factors = cfg.model.encoder.pool_factors
    if any(type(factor) is not int or factor < 1 for factor in factors):
        raise ValueError("encoder.pool_factors must contain positive integers")
    if group % math.prod(factors):
        raise ValueError("mask_group_size must be a multiple of the total encoder pool factor")
    if samples % (patch * group):
        raise ValueError("data.num_samples must be divisible by patch_size * mask_group_size")
    if samples // (patch * group) < 2:
        raise ValueError("data.num_samples must provide at least two mask groups")


def _validate_upstream_forecasting(cfg: DictConfig) -> None:
    """Validate the waveform contract and supported Lightning execution modes."""
    backbone = cfg.model.backbone.config
    metric_interval = cfg.model.get("training_metric_interval", 1)
    if type(metric_interval) is not int or metric_interval < 1:
        raise ValueError("training_metric_interval must be a positive integer")
    waveform_datamodule = cfg.data.get("_target_") in {
        "src.data.wav_datamodule.WAVManifestDataModule",
        "src.data.sharded_audio_datamodule.ShardedAudioDataModule",
    }
    if cfg.data.get("_target_") == "src.data.sharded_audio_datamodule.ShardedAudioDataModule":
        max_steps = cfg.trainer.get("max_steps", -1)
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError(
                "sharded audio requires trainer.max_steps > 0 for its infinite iterator"
            )
    values = {
        "data.num_samples": cfg.data.num_samples,
        "data.sample_rate": cfg.data.sample_rate,
        "backbone.window_samples": backbone.window_samples,
        "backbone.sample_rate": backbone.sample_rate,
        "context_windows": cfg.model.context_windows,
    }
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    horizons, strides = cfg.model.horizons, cfg.model.state_strides
    if len(horizons) != 3 or len(strides) != 3:
        raise ValueError("upstream forecasting requires three horizons and state_strides")
    for name, sequence in (("horizons", horizons), ("state_strides", strides)):
        if any(type(value) is not int or value < 1 for value in sequence):
            raise ValueError(f"{name} must contain positive integers")
    if list(horizons) != sorted(set(horizons)):
        raise ValueError("horizons must be strictly increasing")
    if strides[0] != 1 or list(strides) != sorted(strides):
        raise ValueError("state_strides must start at 1 and be nondecreasing")
    if any((cfg.model.context_windows - 1) % stride for stride in strides):
        raise ValueError("context_windows - 1 must be divisible by every state_stride")
    if any(high % low for low, high in zip(strides[:-1], strides[1:])):
        raise ValueError("state_strides must be nested integer multiples of the prior level")
    aggregation = cfg.model.get("aggregation_windows", [1, 1, 1])
    if (
        len(aggregation) != 3
        or any(type(value) is not int or value < 1 for value in aggregation)
        or aggregation[0] != 1
    ):
        raise ValueError("aggregation_windows must contain three positive integers starting at 1")
    supports = [1]
    for window, previous_stride in zip(aggregation[1:], strides[:-1], strict=True):
        supports.append(supports[-1] + (window - 1) * previous_stride)
    if max(supports) > cfg.model.context_windows:
        raise ValueError("aggregation support must fit the observed context_windows")
    weight = cfg.model.get("temporal_variance_weight", 0.0)
    floor = cfg.model.get("temporal_std_floor", 0.1)
    epsilon = cfg.model.get("temporal_variance_epsilon", 0.0001)
    for name, value in (
        ("temporal_variance_weight", weight),
        ("temporal_std_floor", floor),
        ("temporal_variance_epsilon", epsilon),
    ):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if weight < 0 or epsilon <= 0 or floor <= math.sqrt(epsilon):
        raise ValueError(
            "temporal variance requires nonnegative weight, positive epsilon and "
            "temporal_std_floor > sqrt(temporal_variance_epsilon)"
        )
    if weight > 0:
        if not cfg.model.hierarchy or not all(cfg.model.prediction_weights[1:]):
            raise ValueError("temporal variance requires learned, active upper forecasting spaces")
        if any(cfg.model.context_windows - support + 1 < 2 for support in supports[1:]):
            raise ValueError("temporal variance requires at least two valid upper context states")
    if cfg.data.sample_rate != backbone.sample_rate:
        raise ValueError("data.sample_rate must match backbone.sample_rate")
    crops = cfg.model.get("masked_crops_per_recording", 0)
    crop_samples = cfg.model.get("masked_crop_samples")
    if type(crops) is not int or crops < 0:
        raise ValueError("masked_crops_per_recording must be a nonnegative integer")
    if crops and (
        type(crop_samples) is not int or crop_samples < 1 or cfg.model.masked_weight <= 0
    ):
        raise ValueError("Separate masked crops require a positive length and masked loss")
    if crops != cfg.data.get("masked_crops_per_recording", 0) or crop_samples != cfg.data.get(
        "masked_crop_samples"
    ):
        raise ValueError("Model and data masked crop recipes must match")
    channels = backbone.get("input_channels", 1)
    if channels not in (1, 2):
        raise ValueError("backbone.input_channels must be 1 or 2")
    if cfg.data.get("_target_") == "src.data.sharded_audio_datamodule.ShardedAudioDataModule":
        naturalistic = cfg.data.get("naturalistic_manifest") is not None
        clean_binaural = cfg.data.get("clean_binaural", False)
        if type(clean_binaural) is not bool or (clean_binaural and naturalistic):
            raise ValueError("clean_binaural must be boolean and exclude a naturalistic manifest")
        if (naturalistic or clean_binaural) != (channels == 2):
            raise ValueError(
                "Binaural staged training requires an explicit naturalistic manifest "
                "or clean_binaural=true"
            )
    forecasting = any(cfg.model.prediction_weights) or cfg.model.regularization_weight or weight
    expected = backbone.window_samples * (cfg.model.context_windows + max(horizons))
    if (forecasting or not crops) and cfg.data.num_samples != expected:
        raise ValueError(
            "data.num_samples must equal backbone.window_samples * "
            f"(context_windows + max(horizons)) = {expected}"
        )
    if waveform_datamodule and cfg.data.get("require_full_length") is not True:
        raise ValueError("upstream forecasting requires data.require_full_length=true")
    if cfg.model.regularization_weight > 0:
        if cfg.data.batch_size < 2:
            raise ValueError("regularization requires data.batch_size >= 2")
        if waveform_datamodule and cfg.data.get("minimum_batch_size", 1) < 2:
            raise ValueError(
                "regularization requires data.minimum_batch_size >= 2 for WAV datasets"
            )
    if cfg.trainer.get("accelerator", "cpu") not in ("cpu", "cuda", "gpu"):
        raise ValueError(
            "upstream forecasting supports explicit cpu/cuda/gpu accelerators only; "
            "MPS seeded RNG isolation is not yet validated"
        )
    if cfg.trainer.get("precision", "32-true") not in (32, "32", "32-true", "bf16-mixed"):
        raise ValueError("upstream forecasting supports FP32 or bf16-mixed precision")
    devices = cfg.trainer.get("devices", 1)
    device_count = devices if type(devices) is int and devices > 0 else 0
    if isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices):
        device_count = len(devices)
        if any(type(index) is not int or index < 0 for index in devices):
            device_count = 0
        if len(set(devices)) != len(devices):
            device_count = 0
    nodes = cfg.trainer.get("num_nodes", 1)
    if not device_count or type(nodes) is not int or nodes < 1:
        raise ValueError("upstream forecasting requires explicit positive devices and num_nodes")
    strategy = cfg.trainer.get("strategy", "auto")
    if isinstance(strategy, DictConfig):
        supported = strategy.get("_target_", "") in {
            "lightning.pytorch.strategies.DDPStrategy",
            "lightning.pytorch.strategies.ddp.DDPStrategy",
        }
    else:
        supported = strategy in {
            "auto",
            "ddp",
            "ddp_find_unused_parameters_true",
            "ddp_find_unused_parameters_false",
        }
    if not supported:
        raise ValueError("upstream forecasting supports single-device or DDP strategies only")
    if (
        device_count * nodes > 1
        and waveform_datamodule
        and cfg.trainer.get("use_distributed_sampler", True)
    ):
        raise ValueError(
            "WAV DDP requires trainer.use_distributed_sampler=false; "
            "the datamodule owns rank sharding and minimum batch sizes"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_run(function):
    """Persist local run outcomes, refusing to reuse an existing run directory.

    SIGKILL/power loss cannot run cleanup; a leftover 'running' status is explicitly
    not success. Rank-zero ownership follows Lightning's launcher convention.
    """

    @wraps(function)
    def wrapped(cfg: DictConfig):
        if rank_zero_only.rank != 0:
            return function(cfg)
        output = Path(cfg.paths.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "run.json"
        if not path.exists() and any(
            (output / name).exists() for name in ("provenance.json", "metrics.json", "checkpoints")
        ):
            raise FileExistsError(f"Existing artifacts in {output}; choose a new paths.output_dir")
        record = {
            "schema_version": 1,
            "task": function.__name__,
            "status": "running",
            "started_at": _now(),
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"Run already exists in {output}; choose a new paths.output_dir, including when resuming"
            ) from error
        try:
            metrics, objects = function(cfg)
            serialized = {}
            for name, value in metrics.items():
                scalar = float(
                    value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
                )
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Nonfinite final metric: {name}")
                serialized[name] = scalar
            atomic_json(output / "metrics.json", serialized)
            trainer = objects["trainer"]
            callback = trainer.checkpoint_callback
            checkpoints = {}
            if callback is not None:
                for label, filename in (
                    ("best", callback.best_model_path),
                    ("last", callback.last_model_path),
                ):
                    if filename and Path(filename).is_file():
                        checkpoints[label] = {
                            "path": str(Path(filename).resolve()),
                            "sha256": file_sha256(filename),
                        }
            record.update(
                status="interrupted" if trainer.interrupted else "completed",
                finished_at=_now(),
                global_step=trainer.global_step,
                device=str(trainer.strategy.root_device),
                precision=str(trainer.precision),
                checkpoints=checkpoints,
            )
            atomic_json(path, record)
            return metrics, objects
        except BaseException as error:
            record.update(
                status="interrupted"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "failed",
                finished_at=_now(),
                error_type=type(error).__name__,
                error=str(error),
            )
            atomic_json(path, record)
            raise

    return wrapped

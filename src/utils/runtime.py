"""Small run-safety helpers; Lightning continues to own the training loop."""

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig


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


def validate_config(cfg: DictConfig) -> None:
    """Fail before model allocation for incompatible audio/masking settings."""
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

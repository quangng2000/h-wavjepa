"""One externally launched DDP benchmark using a frozen private AudioSet index.

Example: torchrun --standalone --nproc_per_node=2 scripts/benchmark_streaming_gpu.py
--devices 2 --compact --output /tmp/stream-benchmark

The caller must bound the cloud job's runtime and spending. This script rents no
resources, downloads no model weights, and never launches another process group.
It measures training with real shard refills, then full available validation and
a local checkpoint roundtrip, optionally followed by private-S3 publication and
verified download. A partial index remains explicitly a partial pilot.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import lightning as L
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_gpu import atomic_report  # noqa: E402
from scripts.segment_checkpoint import (  # noqa: E402
    CheckpointRepository,
    checkpoint_client,
    minimum_checkpoint_collective_timeout,
)


def probe_variant(args) -> tuple[bool, bool, int]:
    """Nat is a distinct four-rank, binaural, eight-crop hardware workload."""
    nat = getattr(args, "nat", False)
    manifest = getattr(args, "naturalistic_manifest", None)
    if nat and (args.devices != 4 or manifest is None):
        raise ValueError("Nat requires four GPUs and an explicit --naturalistic-manifest")
    if not nat and manifest is not None:
        raise ValueError("--naturalistic-manifest requires --nat; mono fallback is forbidden")
    return nat, nat or getattr(args, "temporal", False), 16 if nat else 64


def durable_checkpoint_options(args) -> dict | None:
    fields = ("checkpoint_bucket", "checkpoint_prefix", "checkpoint_run_id")
    values = [getattr(args, field, None) for field in fields]
    exclusive = getattr(args, "exclusive_checkpoint_writer", False)
    if not any(values) and not exclusive:
        return None
    if not all(values) or not exclusive:
        raise ValueError(
            "Durable probe requires checkpoint bucket, prefix, run ID and exclusive writer"
        )
    if args.devices != 4 or args.workers != 1:
        raise ValueError("Durable acceptance requires four GPUs and one spawned worker per rank")
    minimum_checkpoint_collective_timeout(args.checkpoint_transfer_timeout_seconds)
    return dict(zip(("bucket", "prefix", "run_id"), values, strict=True))


def collective_timeout_seconds(args) -> int:
    if durable_checkpoint_options(args) is None:
        return 180
    return max(180, minimum_checkpoint_collective_timeout(args.checkpoint_transfer_timeout_seconds))


def open_checkpoint_repository(args, frozen: dict) -> CheckpointRepository | None:
    """Validate separate credentials and an empty owned probe prefix before training."""
    options = durable_checkpoint_options(args)
    if options is None:
        return None
    deadline = time.monotonic() + args.checkpoint_transfer_timeout_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint probe setup exceeded its transfer budget")

    repository = CheckpointRepository(
        checkpoint_client(args.checkpoint_endpoint),
        **options,
        index_sha256=frozen["frozen_index_sha256"],
        global_batch=probe_variant(args)[2],
        check=check,
    )
    repository.open(initialize=True)
    check()
    return repository


def scratch_directory(parent: Path, output: Path) -> Path:
    """Keep audio and full checkpoints out of the platform's /output artifact tree.

    A platform kill bypasses finally blocks. Small reports belong in output;
    disposable shard caches must remain outside it even after that hard kill.
    """
    parent, output = parent.expanduser().resolve(), output.expanduser().resolve()
    for forbidden in (Path("/output").resolve(), output):
        if parent == forbidden or forbidden in parent.parents:
            raise ValueError("Benchmark scratch must be outside /output and the report directory")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="hwavjepa-streaming-", dir=parent))


def worker_facts(loader) -> dict:
    context = loader.multiprocessing_context
    method = context.get_start_method() if context is not None else None
    if loader.num_workers < 1 or method != "spawn":
        raise RuntimeError("The distributed streaming probe requires spawned input workers")
    return {"workers_per_rank": loader.num_workers, "data_worker_start_method": method}


def _cache_inventory(directory: Path, rank: int) -> dict:
    paths = list(directory.glob(f"train/rank-{rank}/*/*.tar"))
    sizes = []
    for path in paths:
        # Other readers may evict a verified object between glob and stat.
        with contextlib.suppress(FileNotFoundError):
            sizes.append(path.stat().st_size)
    return {"verified_files": len(sizes), "verified_bytes": sum(sizes)}


def _nat_asset_catalog(manifest: Path, expected_sha256: str) -> dict:
    """Classify remote cache objects without downloading any BRIR or noise."""
    from src.data.naturalistic_audio import NaturalisticAudio

    renderer = NaturalisticAudio(manifest, require_asset_checksums=True)
    if not renderer.remote_assets or renderer.manifest_sha256 != expected_sha256:
        raise RuntimeError("Nat streaming acceptance requires the pinned remote asset manifest")
    groups = {"noise": set(renderer.noise_files), "brir": set()}
    for scenes in renderer.scenes.values():
        for scene in scenes:
            groups["brir"].update((scene.source_rir, *scene.noise_rirs))
    objects = {}
    for kind, paths in groups.items():
        for path in paths:
            asset = renderer.assets[path]
            value = {"kind": kind, "bytes": asset["bytes"]}
            if asset["sha256"] in objects and objects[asset["sha256"]] != value:
                raise RuntimeError("Nat object identity is ambiguous between BRIR and noise")
            objects[asset["sha256"]] = value
    return {
        "manifest_sha256": renderer.manifest_sha256,
        "bucket": renderer.bucket,
        "endpoint": renderer.endpoint,
        "objects": objects,
    }


def _nat_cache_inventory(directory: Path, rank: int, catalog: dict) -> dict:
    """Inspect only checksum-verified, atomically enrolled remote cache objects.

    _ShardCache publishes these final .tar names only after full SHA-256 readback.
    Comparing object identities, rather than file counts, still detects new
    downloads when a bounded cache evicts an older asset before enrolling one.
    """
    result = {"contracts": 0, "brir": {}, "noise": {}}
    for cache in directory.glob(f"train/rank-{rank}/worker-*/nat"):
        marker = cache / "contract.json"
        if not marker.exists():
            continue
        contract = json.loads(marker.read_text())
        if any(
            contract.get(key) != catalog[key] for key in ("manifest_sha256", "bucket", "endpoint")
        ):
            raise RuntimeError("Nat remote cache contract differs from the pinned manifest")
        result["contracts"] += 1
        for path in cache.glob("*.tar"):
            asset = catalog["objects"].get(path.stem)
            if asset is None or path.is_symlink():
                raise RuntimeError("Unexpected object in Nat's owned remote cache")
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue  # The input worker may evict a verified file during this snapshot.
            if size != asset["bytes"]:
                raise RuntimeError("Nat remote cache object size differs from its manifest")
            identity = f"{cache.parent.name}/{path.stem}"
            result[asset["kind"]][identity] = size
    return result


def _nat_streaming_facts(warm: dict, final: dict, manifest_sha256: str) -> dict:
    def compact(snapshot):
        return {
            "remote_cache_contracts": snapshot["contracts"],
            **{
                kind: {
                    "verified_files": len(snapshot[kind]),
                    "verified_bytes": sum(snapshot[kind].values()),
                }
                for kind in ("brir", "noise")
            },
        }

    return {
        "manifest_sha256": manifest_sha256,
        "warmup": compact(warm),
        "train_end": compact(final),
        "new_verified_brir_objects_after_warmup": len(final["brir"].keys() - warm["brir"].keys()),
        "new_verified_noise_objects_after_warmup": len(
            final["noise"].keys() - warm["noise"].keys()
        ),
    }


def _ear_cnns(model):
    cnns = model.backbone.model.extract_audio.cnns
    if len(cnns) != 2 or not set(p.data_ptr() for p in cnns[0].parameters()).isdisjoint(
        p.data_ptr() for p in cnns[1].parameters()
    ):
        raise RuntimeError("Nat requires two independent ear CNN parameter sets")
    return cnns


def _ear_update_facts(model, initial: list[list[torch.Tensor]]) -> list[dict]:
    facts = []
    for ear, (cnn, saved) in enumerate(zip(_ear_cnns(model), initial, strict=True)):
        parameters = list(cnn.parameters())
        facts.append(
            {
                "ear": ear,
                "independent_parameters": True,
                "parameter_tensors": len(parameters),
                "changed_parameter_tensors": sum(
                    not torch.equal(before, after.detach())
                    for before, after in zip(saved, parameters, strict=True)
                ),
                "all_parameters_finite": bool(
                    torch.stack([torch.isfinite(p).all() for p in parameters]).all()
                ),
            }
        )
    return facts


def _require_nat_training_evidence(report: dict, manifest_sha256: str) -> None:
    assets = report.get("nat_asset_streaming", {})
    ears = report.get("ear_cnn_updates", [])
    if assets.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Nat remote asset streaming evidence is missing or mismatched")
    final = assets.get("train_end", {})
    if final.get("remote_cache_contracts", 0) < 1:
        raise RuntimeError("Nat rank did not access the remote BRIR/noise cache")
    for kind in ("brir", "noise"):
        count = assets.get(f"new_verified_{kind}_objects_after_warmup", 0)
        if (
            type(count) is not int
            or count < 1
            or count > final.get(kind, {}).get("verified_files", 0)
            or final.get(kind, {}).get("verified_bytes", 0) <= 0
        ):
            raise RuntimeError(f"Nat rank did not prove a new remote {kind} download after warmup")
    if len(ears) != 2 or any(
        row.get("ear") != ear
        or row.get("independent_parameters") is not True
        or row.get("all_parameters_finite") is not True
        or not 0 < row.get("changed_parameter_tensors", 0) <= row.get("parameter_tensors", 0)
        for ear, row in enumerate(ears)
    ):
        raise RuntimeError("Nat rank did not prove finite updates to both independent ear CNNs")


def freeze_index(
    output: Path, endpoint: str, *, devices: int, workers: int, global_batch: int = 64
) -> dict:
    from src.data.sharded_audio_datamodule import _read_index, _validate_index

    location = f"s3://{os.environ['HWAVJEPA_INDEX_BUCKET']}/{os.environ['HWAVJEPA_INDEX_KEY']}"
    index, source_sha = _read_index(location, endpoint)
    splits = _validate_index(index, endpoint, allow_partial=True)
    if len(splits["train"]) < devices * workers:
        raise ValueError("The frozen pilot needs at least one training shard per rank/worker")
    validation_clips = sum(shard["splits"]["val"]["clips"] for shard in splits["val"])
    if validation_clips < global_batch:
        raise ValueError("The frozen pilot needs at least one global validation batch")
    path = output / "frozen-index.json"
    atomic_report(path, index)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "path": str(path),
        "source_index_sha256": source_sha,
        "frozen_index_sha256": digest,
        "status": index["status"],
        "verified_sources": index["verified_sources"],
        "split_counts": index["split_counts"],
        "training_shards": len(splits["train"]),
        "validation_clips": validation_clips,
        "validation_used_clips": validation_clips // global_batch * global_batch,
        "validation_excluded_tail_clips": validation_clips % global_batch,
        "global_recording_batch": global_batch,
    }
    atomic_report(output / "frozen-index-receipt.json", summary)
    return summary


def make_config(args, frozen: dict, cache: Path):
    nat, temporal, global_batch = probe_variant(args)
    if nat:
        experiment = "audioset_nat_h100_four"
    else:
        experiment = f"audioset_{'temporal_' if temporal else ''}h100_"
        experiment += "four" if args.devices == 4 else "fast"
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", "data=sharded_audio"],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir = cfg.paths.work_dir = str(ROOT)
    cfg.paths.output_dir = str(args.output / "run")
    cfg.paths.log_dir = str(args.output / "logs")
    cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
    cfg.logger = None
    cfg.callbacks = {
        "timing": {
            "_target_": "scripts.benchmark_streaming_gpu.StreamingTimings",
            "output": str(args.output),
            "cache": str(cache),
            "steps": args.steps,
            "warmup": args.warmup,
            "frozen_sha256": frozen["frozen_index_sha256"],
            "temporal": temporal,
            "nat": nat,
            "naturalistic_sha256": frozen.get("naturalistic_manifest_sha256"),
            "naturalistic_manifest": str(args.naturalistic_manifest) if nat else None,
        }
    }
    cfg.data.index_path = frozen["path"]
    cfg.data.endpoint = args.endpoint
    cfg.data.cache_dir = str(cache)
    cfg.data.cache_max_gib = 40
    cfg.data.batch_size = global_batch // args.devices
    if nat:
        cfg.data.naturalistic_manifest = str(
            Path(args.naturalistic_manifest).expanduser().resolve()
        )
    cfg.data.num_workers = args.workers
    cfg.data.allow_partial_index = True
    cfg.data.allow_stream_restart_on_resume = False
    cfg.trainer.devices = args.devices
    cfg.trainer.max_steps = args.steps
    cfg.trainer.limit_val_batches = 0
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    cfg.trainer.strategy.timeout = {
        "_target_": "datetime.timedelta",
        "seconds": collective_timeout_seconds(args),
    }
    # The Nat preset needs the unbenchmarked capacity-saving path; never silently
    # turn it off while pretending the mono probe measures the same workload.
    if not nat:
        cfg.model.backbone.config.use_gradient_checkpointing = False
        cfg.model.backbone.config.compact_masked_tokens = args.compact
    cfg.ckpt_path, cfg.train, cfg.test = None, True, False
    if (
        cfg.model.lr_total_steps != 375000
        or cfg.model.lr_warmup_steps != 100000
        or cfg.model.ema_anneal_end_step != 100000
    ):
        raise ValueError("Streaming probe must retain the real 375k training schedules")
    return cfg


def architecture_facts(model) -> dict:
    """Read instantiated geometry, not merely the requested command-line flag."""
    return {
        "input_channels": model.backbone.config.get("input_channels", 1),
        "masked_crops_per_recording": model.masked_crops_per_recording,
        "masked_crop_samples": model.masked_crop_samples,
        "aggregation_windows": list(model.aggregation_windows),
        "level_support_windows": list(model.level_support_windows),
        "temporal_variance_weight": model.temporal_variance_weight,
        "temporal_std_floor": model.temporal_std_floor,
        "temporal_variance_epsilon": model.temporal_variance_epsilon,
        "context_windows": model.context_windows,
        "sample_windows": model.sample_windows,
        "state_strides": list(model.state_strides),
        "dense_context_states": [len(p) for p in model.level_positions(model.context_windows)],
        "predictor_context_states": [
            len(p) for p in model.level_positions(model.context_windows, subsample=True)
        ],
        "masked_weight": model.masked_weight,
    }


def require_architecture(facts: dict, *, temporal: bool, nat: bool = False) -> None:
    temporal = temporal or nat
    if not nat:
        # Older passing mono probes predate explicit channel/crop metadata. Only
        # absent fields receive mono defaults; declared Nat values never do.
        facts = {
            "input_channels": 1,
            "masked_crops_per_recording": 0,
            "masked_crop_samples": None,
        } | facts
    expected = {
        "input_channels": 2 if nat else 1,
        "masked_crops_per_recording": 8 if nat else 0,
        "masked_crop_samples": 32160 if nat else None,
        "aggregation_windows": [1, 2, 2] if temporal else [1, 1, 1],
        "level_support_windows": [1, 2, 4] if temporal else [1, 1, 1],
        "temporal_variance_weight": 0.1 if temporal else 0.0,
        "temporal_std_floor": 0.1,
        "temporal_variance_epsilon": 1e-4,
        "context_windows": 5,
        "sample_windows": 9,
        "state_strides": [1, 2, 4],
        "dense_context_states": [5, 4, 2] if temporal else [5, 5, 5],
        "predictor_context_states": [5, 2, 1] if temporal else [5, 3, 2],
        "masked_weight": 1.0,
    }
    if facts != expected:
        raise RuntimeError("Instantiated architecture does not match the requested streaming probe")


def temporal_loss_metrics(metrics: dict, *, stage: str) -> dict:
    suffix = "_step" if stage == "train" else ""
    names = ["temporal_variance_loss"] + [
        f"level_{level}/temporal_variance_loss" for level in (1, 2)
    ]
    try:
        values = {name: float(metrics[f"{stage}/{name}{suffix}"]) for name in names}
    except KeyError as error:
        raise RuntimeError("Temporal-loss evidence is missing") from error
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise RuntimeError("Temporal-loss evidence is negative or nonfinite")
    if not math.isclose(
        values[names[0]], values[names[1]] + values[names[2]], rel_tol=1e-5, abs_tol=1e-7
    ):
        raise RuntimeError("Temporal-loss evidence does not match the upper-level sum")
    return values


class StreamingTimings(L.Callback):
    def __init__(
        self,
        output,
        cache,
        steps,
        warmup,
        frozen_sha256,
        temporal=False,
        nat=False,
        naturalistic_sha256=None,
        naturalistic_manifest=None,
    ):
        self.output, self.cache = Path(output), Path(cache)
        self.steps, self.warmup, self.frozen_sha = steps, warmup, frozen_sha256
        self.nat, self.temporal = nat, temporal or nat
        self.naturalistic_sha256 = naturalistic_sha256
        self.naturalistic_manifest = naturalistic_manifest
        self.global_batch = self.global_masked_crops = None
        self.temporal_samples = []
        self.ends, self.started = [], None
        self.validation_clips = self.validation_batches = 0

    def on_validation_start(self, trainer, pl_module):
        self.validation_clips = self.validation_batches = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.validation_clips += batch["waveform"].shape[0]
        self.validation_batches += 1

    def on_train_start(self, trainer, pl_module):
        self.worker_facts = worker_facts(trainer.train_dataloader)
        self.architecture = architecture_facts(pl_module)
        require_architecture(self.architecture, temporal=self.temporal, nat=self.nat)
        if self.nat and (
            not self.naturalistic_sha256
            or trainer.datamodule.state_dict().get("naturalistic_sha256")
            != self.naturalistic_sha256
        ):
            raise RuntimeError("Nat data module did not use the pinned naturalistic manifest")
        torch.cuda.synchronize()
        self.wall_started = time.perf_counter()
        self.initial = next(pl_module.backbone.model.encoder.parameters()).detach().clone()
        if self.nat:
            self.nat_catalog = _nat_asset_catalog(
                self.naturalistic_manifest, self.naturalistic_sha256
            )
            self.initial_ears = [
                [parameter.detach().clone() for parameter in cnn.parameters()]
                for cnn in _ear_cnns(pl_module)
            ]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        self.global_batch = (
            batch["waveform"].shape[0] * trainer.world_size * trainer.accumulate_grad_batches
        )
        self.global_masked_crops = self.global_batch * pl_module.masked_crops_per_recording
        if self.nat and (
            batch["waveform"].shape[1:] != (2, pl_module.sample_windows * pl_module.window_samples)
            or "masked_crops" not in batch
            or batch["masked_crops"].shape[1:] != (8, 2, 32160)
        ):
            raise RuntimeError("Nat probe requires binaural chronology and eight masked crops")
        if self.temporal and (
            trainer.global_step == 1
            or trainer.global_step % pl_module.training_metric_interval == 0
        ):
            self.temporal_samples.append(
                {
                    "optimizer_step": trainer.global_step,
                    "metrics": temporal_loss_metrics(trainer.callback_metrics, stage="train"),
                }
            )
        if trainer.global_step in (1, self.warmup) or trainer.global_step % 25 == 0:
            atomic_report(
                self.output / f"progress-rank-{trainer.global_rank}.json",
                {
                    **self.worker_facts,
                    "architecture": self.architecture,
                    "temporal_training_samples": self.temporal_samples,
                    "ok": False,
                    "stage": "training_incomplete",
                    "rank": trainer.global_rank,
                    "world_size": trainer.world_size,
                    "optimizer_steps": trainer.global_step,
                    "ema_updates": int(pl_module.ema_updates),
                    "requested_steps": self.steps,
                    "elapsed_seconds_after_first_batch": time.perf_counter() - self.wall_started,
                    "rank_cache": _cache_inventory(self.cache, trainer.global_rank),
                    "frozen_index_sha256": self.frozen_sha,
                },
            )
        if trainer.is_global_zero and trainer.global_step % 100 == 0:
            print(
                f"streaming train progress: step={trainer.global_step}/{self.steps} "
                f"elapsed_seconds={time.perf_counter() - self.wall_started:.1f}",
                flush=True,
            )
        if trainer.global_step == self.warmup:
            self.warm_cache = _cache_inventory(self.cache, trainer.global_rank)
            if self.nat:
                self.warm_nat_cache = _nat_cache_inventory(
                    self.cache, trainer.global_rank, self.nat_catalog
                )
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()
        elif trainer.global_step > self.warmup:
            self.ends.append(time.perf_counter())

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        if trainer.global_step != self.steps or int(pl_module.ema_updates) != self.steps:
            raise RuntimeError("Streaming optimizer/EMA step counts do not match the probe")
        if self.started is None or len(self.ends) != self.steps - self.warmup:
            raise RuntimeError("Incomplete sustained timing interval")
        changed = not torch.equal(
            self.initial, next(pl_module.backbone.model.encoder.parameters()).detach()
        )
        finite = bool(torch.stack([torch.isfinite(p).all() for p in pl_module.parameters()]).all())
        if not changed or not finite:
            raise RuntimeError("Online weights did not change or model parameters became nonfinite")
        intervals = [
            end - start
            for start, end in zip([self.started, *self.ends[:-1]], self.ends, strict=True)
        ]
        end_cache = _cache_inventory(self.cache, trainer.global_rank)
        refills = end_cache["verified_files"] > self.warm_cache["verified_files"]
        report = {
            **self.worker_facts,
            "architecture": self.architecture,
            "temporal_training_samples": self.temporal_samples,
            "rank": trainer.global_rank,
            "world_size": trainer.world_size,
            "global_batch": self.global_batch,
            "global_masked_crops": self.global_masked_crops,
            "naturalistic_manifest_sha256": self.naturalistic_sha256,
            "optimizer_steps": trainer.global_step,
            "ema_updates": int(pl_module.ema_updates),
            "warmup_steps": self.warmup,
            "measured_steps": len(intervals),
            "measured_seconds": self.ends[-1] - self.started,
            "mean_seconds_per_update": statistics.mean(intervals),
            "median_seconds_per_update": statistics.median(intervals),
            "p95_seconds_per_update": sorted(intervals)[int(0.95 * (len(intervals) - 1))],
            "training_wall_seconds_including_warmup": self.ends[-1] - self.wall_started,
            "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            "online_weights_changed": changed,
            "all_model_parameters_finite": finite,
            "input": "frozen private S3 index; production shard cache, WAV decode and random crops",
            "frozen_index_sha256": self.frozen_sha,
            "rank_cache_at_warmup": self.warm_cache,
            "rank_cache_at_train_end": end_cache,
            "cache_refills_observed_after_warmup": refills,
        }
        if self.nat:
            report["nat_asset_streaming"] = _nat_streaming_facts(
                self.warm_nat_cache,
                _nat_cache_inventory(self.cache, trainer.global_rank, self.nat_catalog),
                self.naturalistic_sha256,
            )
            report["ear_cnn_updates"] = _ear_update_facts(pl_module, self.initial_ears)
        atomic_report(self.output / f"rank-{trainer.global_rank}.json", report)


def finite_metrics(results: list[dict]) -> dict:
    if len(results) != 1 or "val/loss" not in results[0]:
        raise RuntimeError("The full validation loop did not return val/loss")
    metrics = {name: float(value) for name, value in results[0].items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Validation returned nonfinite metrics")
    return metrics


def verify_checkpoint(checkpoint, model, datamodule, steps: int) -> dict:
    if (
        checkpoint.get("global_step") != steps
        or int(checkpoint["state_dict"]["ema_updates"]) != steps
    ):
        raise RuntimeError("Saved checkpoint optimizer/EMA counts differ from training")
    if checkpoint.get(type(datamodule).__qualname__) != datamodule.state_dict():
        raise RuntimeError("Saved checkpoint lost the frozen data-index identity")
    if (
        len(checkpoint.get("optimizer_states", [])) != 1
        or len(checkpoint.get("lr_schedulers", [])) != 1
    ):
        raise RuntimeError("Full optimizer/scheduler state was not saved")
    model.on_load_checkpoint(checkpoint)
    expected = model.state_dict()
    saved = checkpoint["state_dict"]
    if set(saved) != set(expected) or any(
        not torch.equal(value.cpu(), saved[name].cpu()) for name, value in expected.items()
    ):
        raise RuntimeError("Saved model tensors differ from the trained model")
    model.load_state_dict(saved, strict=True)
    return architecture_facts(model)


def durable_checkpoint_roundtrip(
    repository: CheckpointRepository,
    checkpoint_path: Path,
    model,
    datamodule,
    steps: int,
    transfer_timeout_seconds: float,
) -> dict:
    """Measure complete publication/readback and independent download verification.

    One cooperative budget covers this entire operation. The outer cloud job and
    its collective timeout remain the backstops for a blocked SDK request.
    """
    minimum_checkpoint_collective_timeout(transfer_timeout_seconds)
    started = time.perf_counter()
    deadline = time.monotonic() + transfer_timeout_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint probe exceeded its transfer budget")

    repository.store.check = check
    check()
    receipt = repository.publish(
        checkpoint_path, steps, segment_id="streaming-probe", data_seed=repository.base_data_seed
    )
    check()
    upload_and_readback_seconds = time.perf_counter() - started
    downloaded = checkpoint_path.with_name("downloaded-checkpoint.ckpt")
    download_started = time.perf_counter()
    try:
        restored_receipt = repository.download_latest(downloaded)
        check()
        if restored_receipt != receipt:
            raise RuntimeError("Checkpoint probe receipt changed before independent download")
        download_seconds = time.perf_counter() - download_started
        verify_started = time.perf_counter()
        restored = torch.load(downloaded, map_location="cpu", weights_only=True)
        architecture = verify_checkpoint(restored, model, datamodule, steps)
        del restored
        check()
        return {
            "upload_and_full_readback_seconds": upload_and_readback_seconds,
            "download_and_sha256_seconds": download_seconds,
            "downloaded_tensor_and_metadata_verify_seconds": time.perf_counter() - verify_started,
            "total_seconds": time.perf_counter() - started,
            "transfer_timeout_seconds": transfer_timeout_seconds,
            "budget_scope": "publication, full remote readback, independent download and verification",
            "complete_remote_readback_verified": True,
            "download_sha256_verified": True,
            "strict_tensor_and_metadata_roundtrip": True,
            "architecture": architecture,
            "receipt": receipt,
        }
    finally:
        downloaded.unlink(missing_ok=True)


def summarize(
    output: Path,
    devices: int,
    steps: int,
    frozen: dict,
    *,
    require_durable_checkpoint: bool = False,
    temporal: bool = False,
    nat: bool = False,
) -> dict:
    temporal = temporal or nat
    expected_batch = 16 if nat else 64
    if nat and devices != 4:
        raise RuntimeError("Nat acceptance requires four GPUs")
    ranks = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(devices)]
    validation = [
        json.loads((output / f"validation-rank-{rank}.json").read_text()) for rank in range(devices)
    ]
    for rank, (train, val) in enumerate(zip(ranks, validation, strict=True)):
        if any(
            row.get("rank") != rank
            or row.get("world_size") != devices
            or row.get("optimizer_steps") != steps
            or row.get("ema_updates") != steps
            or row.get("frozen_index_sha256") != frozen["frozen_index_sha256"]
            for row in (train, val)
        ):
            raise RuntimeError("Rank reports disagree on topology, progress or frozen dataset")
        if not train.get("online_weights_changed") or not train.get("all_model_parameters_finite"):
            raise RuntimeError("A rank did not prove finite parameter updates")
        if train.get("global_batch") != expected_batch:
            raise RuntimeError("Rank reports a different global recording batch")
        if nat and (
            train.get("global_masked_crops") != 128
            or not frozen.get("naturalistic_manifest_sha256")
            or any(
                row.get("naturalistic_manifest_sha256") != frozen["naturalistic_manifest_sha256"]
                for row in (train, val)
            )
        ):
            raise RuntimeError("Nat rank crops or naturalistic manifest identity differ")
        if nat:
            _require_nat_training_evidence(train, frozen["naturalistic_manifest_sha256"])
        for row in (train, val):
            require_architecture(row.get("architecture", {}), temporal=temporal, nat=nat)
        if temporal:
            samples = train.get("temporal_training_samples", [])
            if not samples or samples[0].get("optimizer_step") != 1:
                raise RuntimeError("A rank did not report sampled temporal training losses")
            for sample in samples:
                if not 1 <= sample.get("optimizer_step", 0) <= steps:
                    raise RuntimeError("Temporal-loss sample falls outside the training interval")
                temporal_loss_metrics(
                    {f"train/{name}_step": value for name, value in sample["metrics"].items()},
                    stage="train",
                )
            temporal_loss_metrics(val["metrics"], stage="val")
        if (
            train.get("data_worker_start_method") != "spawn"
            or train.get("workers_per_rank", 0) < 1
            or train.get("workers_per_rank") != ranks[0].get("workers_per_rank")
        ):
            raise RuntimeError("Every rank must prove the same spawned input-worker setup")
        if (
            not train.get("cache_refills_observed_after_warmup")
            or train["rank_cache_at_train_end"]["verified_files"]
            <= train["rank_cache_at_warmup"]["verified_files"]
        ):
            raise RuntimeError(
                "Every rank must show additional verified shard downloads after warmup"
            )
        finite_metrics([val["metrics"]])
        if val.get("clips") != frozen["validation_used_clips"] // devices:
            raise RuntimeError("A rank did not complete its full validation allocation")
    seconds = max(row["mean_seconds_per_update"] for row in ranks)
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Invalid sustained training duration")
    checkpoint = json.loads((output / "checkpoint.json").read_text())
    if (
        checkpoint.get("strict_tensor_and_metadata_roundtrip") is not True
        or checkpoint.get("bytes", 0) <= 0
    ):
        raise RuntimeError("Checkpoint roundtrip evidence is missing")
    require_architecture(checkpoint.get("architecture", {}), temporal=temporal, nat=nat)
    if require_durable_checkpoint:
        durable = checkpoint.get("durable_s3", {})
        receipt = durable.get("receipt", {})
        if (
            devices != 4
            or ranks[0]["workers_per_rank"] != 1
            or any(
                durable.get(field) is not True
                for field in (
                    "complete_remote_readback_verified",
                    "download_sha256_verified",
                    "strict_tensor_and_metadata_roundtrip",
                )
            )
            or receipt.get("owner", {}).get("index_sha256") != frozen["frozen_index_sha256"]
            or receipt.get("owner", {}).get("global_batch") != expected_batch
            or receipt.get("checkpoint", {}).get("step") != steps
            or receipt.get("checkpoint", {}).get("bytes") != checkpoint["bytes"]
        ):
            raise RuntimeError("Durable S3 checkpoint roundtrip evidence is missing or mismatched")
        require_architecture(durable.get("architecture", {}), temporal=temporal, nat=nat)
    return {
        "ok": True,
        "model_variant": "nat_temporal_aggregation"
        if nat
        else ("temporal_aggregation" if temporal else "pointwise"),
        "global_recording_batch": expected_batch,
        "global_masked_crops": 128 if nat else 0,
        "architecture": ranks[0]["architecture"],
        "data_worker_start_method": "spawn",
        "workers_per_rank": ranks[0]["workers_per_rank"],
        "input": "private S3 streaming from a frozen index",
        "frozen_index": frozen,
        "ranks": ranks,
        "validation": validation,
        "seconds_per_update": seconds,
        "updates_per_second": 1 / seconds,
        "projected_375k_train_hours": seconds * 375000 / 3600,
        "standard_batch_gpu_cost_usd": seconds * 375000 / 60 * devices * 0.063,
        "cache_refills_observed_on_all_ranks": True,
        "checkpoint": checkpoint,
        "durable_checkpoint_required": require_durable_checkpoint,
        "limitations": [
            "A passing execution probe does not establish temporal representation quality or rule out collapse.",
            "Only the frozen verified cohort was sampled; partial staging is not the full dataset.",
            "Short measured cache refills cannot guarantee long-run storage or GPU throughput.",
            "Steady-state extrapolation excludes validation, checkpoint uploads and queue time.",
            "A checkpoint tensor/metadata roundtrip does not prove resumed stream-position recovery.",
        ],
    }


def run(args):
    from src.train import train

    checkpoint_required = durable_checkpoint_options(args) is not None
    nat, temporal, global_batch = probe_variant(args)
    collective_seconds = collective_timeout_seconds(args)
    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != args.devices or torch.cuda.device_count() < args.devices:
        raise ValueError("Launch with torchrun and the requested two or four visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The streaming benchmark requires BF16-capable CUDA hardware")
    dist.init_process_group("nccl", timeout=timedelta(seconds=collective_seconds))
    setup = [None]
    repository = None
    if rank == 0:
        scratch = None
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            scratch = scratch_directory(args.scratch_parent, args.output)
            setup[0] = {
                "scratch": str(scratch),
                "frozen": freeze_index(
                    args.output,
                    args.endpoint,
                    devices=world,
                    workers=args.workers,
                    global_batch=global_batch,
                ),
            }
            if nat:
                manifest = Path(args.naturalistic_manifest).expanduser().resolve()
                setup[0]["frozen"]["naturalistic_manifest_sha256"] = hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest()
                atomic_report(args.output / "frozen-index-receipt.json", setup[0]["frozen"])
            repository = open_checkpoint_repository(args, setup[0]["frozen"])
        except Exception as error:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
            setup[0] = {"error": type(error).__name__}
    dist.broadcast_object_list(setup, src=0)
    if "error" in setup[0]:
        dist.destroy_process_group()
        raise RuntimeError(f"Frozen dataset setup failed: {setup[0]['error']}")
    frozen = setup[0]["frozen"]
    scratch = Path(setup[0]["scratch"])
    cache, checkpoint_path = scratch / "audio-cache", scratch / "checkpoint.ckpt"
    if rank == 0:
        print(
            f"streaming index frozen: status={frozen['status']} "
            f"verified_sources={frozen['verified_sources']} "
            f"training_shards={frozen['training_shards']} "
            f"validation_clips={frozen['validation_clips']} "
            f"sha256={frozen['frozen_index_sha256']}",
            flush=True,
        )
    try:
        if (
            hashlib.sha256(Path(frozen["path"]).read_bytes()).hexdigest()
            != frozen["frozen_index_sha256"]
        ):
            raise RuntimeError("A rank read a different frozen index")
        fit_started = time.perf_counter()
        if rank == 0:
            print(
                f"streaming train start: devices={world} global_batch={global_batch} "
                f"steps={args.steps} warmup={args.warmup} workers_per_rank={args.workers} "
                f"compact_masked_tokens={True if nat else args.compact} temporal_aggregation={temporal} nat={nat}",
                flush=True,
            )
        _, objects = train(make_config(args, frozen, cache))
        fit_wall = time.perf_counter() - fit_started
        report_path = args.output / f"rank-{rank}.json"
        report = json.loads(report_path.read_text())
        report["setup_and_fit_wall_seconds_including_initial_loading"] = fit_wall
        atomic_report(report_path, report)
        trainer, model, datamodule = (objects[key] for key in ("trainer", "model", "datamodule"))
        dist.barrier()
        validation_started = time.perf_counter()
        if rank == 0:
            print(
                f"streaming validation start: global_clips={frozen['validation_used_clips']}",
                flush=True,
            )
        trainer.limit_val_batches = 1.0
        metrics = finite_metrics(
            trainer.validate(model=model, datamodule=datamodule, verbose=False)
        )
        if temporal:
            temporal_loss_metrics(metrics, stage="val")
        timing = objects["callbacks"][0]
        if timing.validation_clips != frozen["validation_used_clips"] // world:
            raise RuntimeError("Validation did not consume the complete frozen held-out allocation")
        torch.cuda.synchronize()
        dist.barrier()
        atomic_report(
            args.output / f"validation-rank-{rank}.json",
            {
                "rank": rank,
                "architecture": architecture_facts(model),
                "world_size": world,
                "optimizer_steps": trainer.global_step,
                "ema_updates": int(model.ema_updates),
                "frozen_index_sha256": frozen["frozen_index_sha256"],
                "naturalistic_manifest_sha256": datamodule.state_dict().get("naturalistic_sha256"),
                "seconds": time.perf_counter() - validation_started,
                "metrics": metrics,
                "clips": timing.validation_clips,
                "batches": timing.validation_batches,
                "global_used_clips": frozen["validation_used_clips"],
                "global_excluded_tail_clips": frozen["validation_excluded_tail_clips"],
            },
        )
        dist.barrier()
        saved_at = time.perf_counter()
        if rank == 0:
            print("streaming checkpoint save and verification start", flush=True)
        trainer.save_checkpoint(checkpoint_path, weights_only=False)
        dist.barrier()
        checkpoint_outcome = [None]
        if rank == 0:
            try:
                saved_seconds = time.perf_counter() - saved_at
                loaded_at = time.perf_counter()
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                architecture = verify_checkpoint(checkpoint, model, datamodule, args.steps)
                checkpoint_report = {
                    "save_seconds": saved_seconds,
                    "load_verify_seconds": time.perf_counter() - loaded_at,
                    "bytes": checkpoint_path.stat().st_size,
                    "strict_tensor_and_metadata_roundtrip": True,
                    "architecture": architecture,
                    "stream_positions_checkpointed": False,
                }
                del checkpoint
                if repository is not None:
                    print("streaming durable checkpoint publication and download start", flush=True)
                    atomic_report(
                        args.output / "checkpoint-progress.json",
                        {"ok": False, "stage": "durable_checkpoint_incomplete"},
                    )
                    checkpoint_report["durable_s3"] = durable_checkpoint_roundtrip(
                        repository,
                        checkpoint_path,
                        model,
                        datamodule,
                        args.steps,
                        args.checkpoint_transfer_timeout_seconds,
                    )
                atomic_report(args.output / "checkpoint.json", checkpoint_report)
                checkpoint_path.unlink()
                checkpoint_outcome[0] = {"ok": True}
            except Exception as error:
                # SDK exceptions can include signed URLs; only share their class.
                checkpoint_outcome[0] = {"error": type(error).__name__}
                atomic_report(
                    args.output / "checkpoint-failure.json",
                    {"ok": False, "stage": "checkpoint", **checkpoint_outcome[0]},
                )
        dist.broadcast_object_list(checkpoint_outcome, src=0)
        if "error" in checkpoint_outcome[0]:
            raise RuntimeError(
                f"Streaming checkpoint verification failed ({checkpoint_outcome[0]['error']})"
            )
        if rank == 0:
            summary = summarize(
                args.output,
                world,
                args.steps,
                frozen,
                require_durable_checkpoint=checkpoint_required,
                temporal=temporal,
                nat=nat,
            )
            summary["compact_masked_tokens"] = model.backbone.config["compact_masked_tokens"]
            summary["gradient_checkpointing"] = model.backbone.config["use_gradient_checkpointing"]
            summary["workers_per_rank"] = args.workers
            summary["hardware"] = {
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(device) for device in range(world)],
            }
            atomic_report(args.output / "streaming-benchmark-success.json", summary)
            if os.environ.get("GMN_RESULT_PATH"):
                atomic_report(Path(os.environ["GMN_RESULT_PATH"]), summary)
            print(json.dumps(summary), flush=True)
        dist.barrier()
    finally:
        if rank == 0:
            shutil.rmtree(scratch, ignore_errors=True)
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use the temporal AudioSet preset: causal 1/2/4-window support and a 0.1 variance weight",
    )
    parser.add_argument(
        "--nat", action="store_true", help="Probe WavJEPA-Nat temporal training on four GPUs"
    )
    parser.add_argument(
        "--naturalistic-manifest",
        type=Path,
        help="Pinned local binaural scene/noise asset manifest",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Updates to measure (default: 400 mono, 600 Nat so four recordings/rank refill shards)",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scratch-parent",
        type=Path,
        default=Path("/tmp"),
        help="Disposable cache/checkpoint root, outside /output",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HWAVJEPA_S3_ENDPOINT", "https://storage.givemeanode.com"),
    )
    parser.add_argument("--checkpoint-bucket")
    parser.add_argument("--checkpoint-prefix")
    parser.add_argument("--checkpoint-run-id")
    parser.add_argument("--exclusive-checkpoint-writer", action="store_true")
    parser.add_argument("--checkpoint-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument(
        "--checkpoint-transfer-timeout-seconds",
        type=float,
        default=600,
        help="Combined publication, full readback, download and strict verification budget",
    )
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 600 if args.nat else 400
    if not 1 <= args.warmup < args.steps <= 1000 or not 1 <= args.workers <= 4:
        parser.error("Require 1 <= warmup < steps <= 1000 and 1 <= workers <= 4")
    try:
        probe_variant(args)
        durable_checkpoint_options(args)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.expanduser().resolve()
    run(args)


if __name__ == "__main__":
    main()

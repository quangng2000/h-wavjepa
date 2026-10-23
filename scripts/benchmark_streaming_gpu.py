"""One externally launched DDP benchmark using a frozen private AudioSet index.

Example: torchrun --standalone --nproc_per_node=2 scripts/benchmark_streaming_gpu.py
--devices 2 --compact --output /tmp/stream-benchmark

The caller must bound the cloud job's runtime and spending. This script rents no
resources, downloads no model weights, and never launches another process group.
It measures training with real shard refills, then full available validation and
a local checkpoint roundtrip. A partial index remains explicitly a partial pilot.
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


def _cache_inventory(directory: Path, rank: int) -> dict:
    paths = list(directory.glob(f"train/rank-{rank}/*/*.tar"))
    sizes = []
    for path in paths:
        # Other readers may evict a verified object between glob and stat.
        with contextlib.suppress(FileNotFoundError):
            sizes.append(path.stat().st_size)
    return {"verified_files": len(sizes), "verified_bytes": sum(sizes)}


def freeze_index(output: Path, endpoint: str, *, devices: int, workers: int) -> dict:
    from src.data.sharded_audio_datamodule import _read_index, _validate_index

    location = f"s3://{os.environ['HWAVJEPA_INDEX_BUCKET']}/{os.environ['HWAVJEPA_INDEX_KEY']}"
    index, source_sha = _read_index(location, endpoint)
    splits = _validate_index(index, endpoint, allow_partial=True)
    if len(splits["train"]) < devices * workers:
        raise ValueError("The frozen pilot needs at least one training shard per rank/worker")
    validation_clips = sum(shard["splits"]["val"]["clips"] for shard in splits["val"])
    if validation_clips < 64:
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
        "validation_used_clips": validation_clips // 64 * 64,
        "validation_excluded_tail_clips": validation_clips % 64,
    }
    atomic_report(output / "frozen-index-receipt.json", summary)
    return summary


def make_config(args, frozen: dict, cache: Path):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train",
            overrides=["experiment=audioset_h100_fast", "data=sharded_audio"],
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
        }
    }
    cfg.data.index_path = frozen["path"]
    cfg.data.endpoint = args.endpoint
    cfg.data.cache_dir = str(cache)
    cfg.data.cache_max_gib = 40
    cfg.data.batch_size = 64 // args.devices
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


class StreamingTimings(L.Callback):
    def __init__(self, output, cache, steps, warmup, frozen_sha256):
        self.output, self.cache = Path(output), Path(cache)
        self.steps, self.warmup, self.frozen_sha = steps, warmup, frozen_sha256
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
        torch.cuda.synchronize()
        self.wall_started = time.perf_counter()
        self.initial = next(pl_module.backbone.model.encoder.parameters()).detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        if trainer.is_global_zero and trainer.global_step % 100 == 0:
            print(
                f"streaming train progress: step={trainer.global_step}/{self.steps} "
                f"elapsed_seconds={time.perf_counter() - self.wall_started:.1f}",
                flush=True,
            )
        if trainer.global_step == self.warmup:
            self.warm_cache = _cache_inventory(self.cache, trainer.global_rank)
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
            "rank": trainer.global_rank,
            "world_size": trainer.world_size,
            "global_batch": 64,
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
        atomic_report(self.output / f"rank-{trainer.global_rank}.json", report)


def finite_metrics(results: list[dict]) -> dict:
    if len(results) != 1 or "val/loss" not in results[0]:
        raise RuntimeError("The full validation loop did not return val/loss")
    metrics = {name: float(value) for name, value in results[0].items()}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Validation returned nonfinite metrics")
    return metrics


def verify_checkpoint(checkpoint, model, datamodule, steps: int) -> None:
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


def summarize(output: Path, devices: int, steps: int, frozen: dict) -> dict:
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
    return {
        "ok": True,
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
        "limitations": [
            "Only the frozen verified cohort was sampled; partial staging is not the full dataset.",
            "Short measured cache refills cannot guarantee long-run storage or GPU throughput.",
            "Steady-state extrapolation excludes validation, checkpoint uploads and queue time.",
            "A checkpoint tensor/metadata roundtrip does not prove resumed stream-position recovery.",
        ],
    }


def run(args):
    from src.train import train

    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != args.devices or torch.cuda.device_count() < args.devices:
        raise ValueError("Launch with torchrun and the requested two or four visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The streaming benchmark requires BF16-capable CUDA hardware")
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    cache = args.output.parent / f"{args.output.name}-temporary-audio-cache"
    checkpoint_path = args.output.parent / f"{args.output.name}-temporary.ckpt"
    setup = [None]
    if rank == 0:
        owns_cache = False
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            if checkpoint_path.exists():
                raise FileExistsError("Refusing to replace an existing temporary checkpoint")
            cache.mkdir(exist_ok=False)
            owns_cache = True
            setup[0] = {
                "frozen": freeze_index(
                    args.output, args.endpoint, devices=world, workers=args.workers
                )
            }
        except Exception as error:
            if owns_cache:
                shutil.rmtree(cache, ignore_errors=True)
            setup[0] = {"error": type(error).__name__}
    dist.broadcast_object_list(setup, src=0)
    if "error" in setup[0]:
        dist.destroy_process_group()
        raise RuntimeError(f"Frozen dataset setup failed: {setup[0]['error']}")
    frozen = setup[0]["frozen"]
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
                f"streaming train start: devices={world} global_batch=64 "
                f"steps={args.steps} warmup={args.warmup} workers_per_rank={args.workers} "
                f"compact_masked_tokens={args.compact}",
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
        timing = objects["callbacks"][0]
        if timing.validation_clips != frozen["validation_used_clips"] // world:
            raise RuntimeError("Validation did not consume the complete frozen held-out allocation")
        torch.cuda.synchronize()
        dist.barrier()
        atomic_report(
            args.output / f"validation-rank-{rank}.json",
            {
                "rank": rank,
                "world_size": world,
                "optimizer_steps": trainer.global_step,
                "ema_updates": int(model.ema_updates),
                "frozen_index_sha256": frozen["frozen_index_sha256"],
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
        if rank == 0:
            saved_seconds = time.perf_counter() - saved_at
            loaded_at = time.perf_counter()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            verify_checkpoint(checkpoint, model, datamodule, args.steps)
            atomic_report(
                args.output / "checkpoint.json",
                {
                    "save_seconds": saved_seconds,
                    "load_verify_seconds": time.perf_counter() - loaded_at,
                    "bytes": checkpoint_path.stat().st_size,
                    "strict_tensor_and_metadata_roundtrip": True,
                    "stream_positions_checkpointed": False,
                },
            )
            del checkpoint
            checkpoint_path.unlink()
        dist.barrier()
        if rank == 0:
            summary = summarize(args.output, world, args.steps, frozen)
            summary["compact_masked_tokens"] = args.compact
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
            checkpoint_path.unlink(missing_ok=True)
            shutil.rmtree(cache, ignore_errors=True)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, choices=(2, 4), required=True)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HWAVJEPA_S3_ENDPOINT", "https://storage.givemeanode.com"),
    )
    args = parser.parse_args()
    if not 1 <= args.warmup < args.steps <= 1000 or not 1 <= args.workers <= 4:
        parser.error("Require 1 <= warmup < steps <= 1000 and 1 <= workers <= 4")
    args.output = args.output.expanduser().resolve()
    run(args)


if __name__ == "__main__":
    main()

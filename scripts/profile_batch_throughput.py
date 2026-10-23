"""One bounded eight-GPU batch-size profile; no checkpoint writes or publication.

Each invocation launches a fresh torchrun group, restores a pinned full-state
checkpoint in memory, and discards its updates. Input is the real frozen AudioSet
stream using the retained node-local cache and the same data seed per candidate.
This is a short capacity/throughput profile, not streaming or scientific acceptance.
Run only after the production training process and its readers have exited.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightning as L  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from src.utils.runtime import atomic_json, file_sha256  # noqa: E402

CANDIDATES = (16, 32, 64, 128)


def source_base_seed(source):
    owned = source.get("repository_options", {}).get("base_data_seed")
    base = source.get("base_data_seed", owned)
    if (
        type(base) is not int
        or not 0 <= base < 2**32 - 375000
        or (owned is not None and (type(owned) is not int or owned != base))
    ):
        raise ValueError("Source plan requires a consistent owned base data seed")
    return base


def validate_profile(value):
    if type(value.get("global_batch")) is not int or value["global_batch"] not in CANDIDATES:
        raise ValueError("Global batch must be one of 16, 32, 64, 128")
    for name, low, high in (("warmup", 1, 10), ("measured", 1, 50), ("timeout_seconds", 30, 600)):
        if type(value.get(name)) is not int or not low <= value[name] <= high:
            raise ValueError(f"Invalid bounded profile {name}")
    if (
        type(value.get("resume_step")) is not int
        or not 0 < value["resume_step"] < 375000 - value["warmup"] - value["measured"]
    ):
        raise ValueError("Profile requires a nonfinal optimizer step from a full checkpoint")
    digest = value.get("checkpoint_sha256", "")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Profile requires the pinned checkpoint SHA256")
    source = value["source_plan"]
    source_base_seed(source)
    if source.get("clean_binaural") is not True or source.get("nat") is not False:
        raise ValueError("This profile supports the active clean-binaural temporal recipe only")
    topology = source.get("distributed", {})
    if (topology.get("nnodes"), topology.get("nproc_per_node"), topology.get("node_rank")) != (
        1,
        8,
        0,
    ):
        raise ValueError("Profile requires an existing one-node/eight-rank source plan")


def profile_config(value):
    from scripts.train_segment import make_config

    validate_profile(value)
    plan = deepcopy(value["source_plan"])
    plan.update(
        resume_path=value["checkpoint"],
        resume_step=value["resume_step"],
        resolved_data_seed=source_base_seed(plan) + value["resume_step"],
        activation_checkpointing="off",
        shard_prefetch=1,
        global_batch=value["global_batch"],
    )
    cfg = make_config(plan)
    cfg.callbacks = {
        "timing": {
            "_target_": "scripts.profile_batch_throughput.ProfileTiming",
            "profile": value,
        }
    }
    cfg.logger = None
    cfg.paths.output_dir = str(Path(value["output"]) / "run")
    cfg.paths.log_dir = str(Path(value["output"]) / "logs")
    cfg.trainer.max_steps = value["resume_step"] + value["warmup"] + value["measured"]
    cfg.trainer.limit_val_batches = 0
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    # A cold rank may need the full 300s download / 360s worker allowance.
    # Let the supervisor's default 360s wall cap stop and reap the whole group,
    # rather than having healthy peers abort a collective during a slow fetch.
    cfg.trainer.strategy.timeout = {"_target_": "datetime.timedelta", "seconds": 420}
    cfg.train, cfg.test = True, False
    return cfg


class ProfileTiming(L.Callback):
    def __init__(self, profile):
        self.profile = dict(profile)
        self.started = None
        self.ends = []

    def on_train_start(self, trainer, model):
        value = self.profile
        if (
            trainer.world_size != 8
            or not torch.cuda.is_bf16_supported()
            or trainer.global_step != value["resume_step"]
            or int(model.ema_updates) != value["resume_step"]
            or not trainer.optimizers[0].state
            or not self.optimizer_steps_match(trainer, value["resume_step"])
            or trainer.lr_scheduler_configs[0].scheduler.last_epoch != value["resume_step"]
        ):
            raise RuntimeError(
                "Profile did not restore the complete eight-rank optimizer/EMA schedule"
            )
        parameter = next(p for p in model.backbone.model.encoder.parameters() if p.requires_grad)
        self.parameter, self.initial = parameter, parameter.detach().clone()

    @staticmethod
    def optimizer_steps_match(trainer, expected):
        states = trainer.optimizers[0].state.values()
        return bool(trainer.optimizers[0].state) and all(
            isinstance(state, dict) and int(state.get("step", -1)) == expected for state in states
        )

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        completed = trainer.global_step - self.profile["resume_step"]
        if completed < self.profile["warmup"]:
            return
        torch.cuda.synchronize()
        if completed == self.profile["warmup"]:
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()
        else:
            self.ends.append(time.perf_counter())

    def on_train_end(self, trainer, model):
        value = self.profile
        torch.cuda.synchronize()
        if (
            trainer.global_step != value["resume_step"] + value["warmup"] + value["measured"]
            or int(model.ema_updates) != trainer.global_step
            or trainer.lr_scheduler_configs[0].scheduler.last_epoch != trainer.global_step
            or not self.optimizer_steps_match(trainer, trainer.global_step)
            or self.started is None
            or len(self.ends) != value["measured"]
        ):
            raise RuntimeError("Profile did not finish its bounded optimizer and EMA updates")
        finite = torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
        changed = not torch.equal(self.initial, self.parameter.detach())
        if not bool(finite) or not changed:
            raise RuntimeError("Profile requires finite updated model parameters")
        intervals = [
            end - start
            for start, end in zip([self.started, *self.ends[:-1]], self.ends, strict=True)
        ]
        report = {
            "rank": trainer.global_rank,
            "world_size": trainer.world_size,
            "global_batch": value["global_batch"],
            "local_batch": value["global_batch"] // 8,
            "resume_step": value["resume_step"],
            "final_step": trainer.global_step,
            "ema_updates": int(model.ema_updates),
            "scheduler_step": trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            "optimizer_steps_match": True,
            "warmup_steps": value["warmup"],
            "measured_steps": len(intervals),
            "measured_seconds": self.ends[-1] - self.started,
            "median_seconds_per_update": statistics.median(intervals),
            "p95_seconds_per_update": sorted(intervals)[int(0.95 * (len(intervals) - 1))],
            "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            "finite_parameters": True,
            "online_weights_changed": True,
            "checkpoint_sha256": value["checkpoint_sha256"],
            "checkpoint_written": False,
        }
        atomic_json(Path(value["output"]) / f"rank-{trainer.global_rank}.json", report)


def summarize(value, reports):
    if sorted(row["rank"] for row in reports) != list(range(8)):
        raise ValueError("Profile needs eight distinct completed rank reports")
    for row in reports:
        if (
            row.get("world_size") != 8
            or row.get("global_batch") != value["global_batch"]
            or row.get("local_batch") != value["global_batch"] // 8
            or row.get("measured_steps") != value["measured"]
            or row.get("warmup_steps") != value["warmup"]
            or row.get("resume_step") != value["resume_step"]
            or row.get("final_step") != value["resume_step"] + value["warmup"] + value["measured"]
            or row.get("ema_updates") != row.get("final_step")
            or row.get("scheduler_step") != row.get("final_step")
            or row.get("optimizer_steps_match") is not True
            or row.get("checkpoint_sha256") != value["checkpoint_sha256"]
            or row.get("finite_parameters") is not True
            or row.get("online_weights_changed") is not True
            or row.get("checkpoint_written") is not False
            or not math.isfinite(row.get("measured_seconds", float("nan")))
            or row["measured_seconds"] <= 0
        ):
            raise ValueError("Incomplete or inconsistent profile rank evidence")
        for key in (
            "allocated_peak_gib",
            "reserved_peak_gib",
            "median_seconds_per_update",
            "p95_seconds_per_update",
        ):
            number = row.get(key)
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ValueError("Incomplete or inconsistent profile rank evidence")
    elapsed = max(row["measured_seconds"] for row in reports)
    updates = value["measured"] / elapsed
    return {
        "status": "complete",
        "profile_only": True,
        "global_batch": value["global_batch"],
        "optimizer_updates_per_second": updates,
        "recordings_per_second": updates * value["global_batch"],
        "masked_crops_per_second": updates * value["global_batch"] * 8,
        "measured_seconds": elapsed,
        "allocated_peak_gib": max(row["allocated_peak_gib"] for row in reports),
        "reserved_peak_gib": max(row["reserved_peak_gib"] for row in reports),
        "rank_reports": reports,
        "checkpoint_written": False,
        "scope": "Short real-stream profile; cached shards may hide later refill stalls. Updates discarded.",
        "research_caveat": "Larger batches change regularizer population and examples per optimizer update.",
    }


def run_candidate(value):
    from scripts.train_segment import remember_descendants, stop_process_tree

    validate_profile(value)
    # Do not initialize a CUDA context on the supervisor's default GPU: each
    # worker checks BF16 after Lightning has selected its local device.
    if torch.cuda.device_count() != 8:
        raise RuntimeError("Profile requires eight visible CUDA GPUs")
    checkpoint = Path(value["checkpoint"])
    if not checkpoint.is_file() or file_sha256(checkpoint) != value["checkpoint_sha256"]:
        raise ValueError("Profile checkpoint differs from its pinned SHA256")
    output = Path(value["output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "profile-plan.json", value)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "--max_restarts=0",
        str(Path(__file__).resolve()),
        "--worker-plan",
        str(output / "profile-plan.json"),
    ]
    process, owned = None, {}
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        import psutil

        root = psutil.Process(process.pid)
        owned[(root.pid, root.create_time())] = root
        while process.poll() is None:
            remember_descendants(owned)
            if time.monotonic() - started >= value["timeout_seconds"]:
                raise TimeoutError("Candidate wall-clock limit reached")
            time.sleep(0.1)
        remember_descendants(owned)
        if process.returncode:
            raise RuntimeError(
                f"Candidate torchrun exit code {process.returncode}; inspect bounded log"
            )
        reports = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(8)]
        summary = summarize(value, reports)
        # Verify every owned worker/helper exited before declaring success.
        stop_process_tree(process, owned, grace_seconds=2)
        process = None
        summary["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "profile-result.json", summary)
        print(
            json.dumps({key: item for key, item in summary.items() if key != "rank_reports"}),
            flush=True,
        )
    except BaseException as error:
        atomic_json(
            output / "profile-result.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "global_batch": value["global_batch"],
                "wall_seconds": time.monotonic() - started,
                "checkpoint_written": False,
            },
        )
        raise
    finally:
        if process is not None:
            stop_process_tree(process, owned, grace_seconds=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--global-batch", type=int, choices=CANDIDATES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_plan:
        if int(os.environ.get("WORLD_SIZE", "0")) != 8:
            raise ValueError("Profile worker requires eight externally launched ranks")
        from src.train import train

        train(profile_config(json.loads(args.worker_plan.read_text())))
        return
    if any(
        getattr(args, field) is None
        for field in (
            "source_plan",
            "checkpoint",
            "checkpoint_sha256",
            "resume_step",
            "global_batch",
            "output",
        )
    ):
        parser.error(
            "Source plan, pinned checkpoint/SHA/step, batch and new output path are required"
        )
    value = {
        "source_plan": json.loads(args.source_plan.read_text()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "resume_step": args.resume_step,
        "global_batch": args.global_batch,
        "warmup": args.warmup,
        "measured": args.measured,
        "timeout_seconds": args.timeout_seconds,
        "output": str(args.output.resolve()),
    }
    run_candidate(value)


if __name__ == "__main__":
    main()

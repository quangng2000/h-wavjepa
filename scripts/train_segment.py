"""Run one bounded four-H100 segment; never rent, restart, or top up compute.

The orchestrator must check the organization's balance before EACH launch and
submit the printed platform limit with max_restarts=0. This process cannot read
account balances using bucket credentials. --plan-only does no network I/O.

Use a frozen, completed staging index and a successful streaming GPU report for
the spawn-worker path. Resume restores full Lightning state but restarts data
iterators; --acknowledge-stream-restart makes this limitation explicit.
The data seed is base_data_seed + resumed optimizer step; the model seed and
LR/EMA schedules stay fixed. This changes restart order, not sample-exact resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.segment_checkpoint import (  # noqa: E402
    SHA,
    CheckpointRepository,
    checkpoint_client,
)
from src.utils.runtime import atomic_json, file_sha256  # noqa: E402


@dataclass(frozen=True)
class SegmentBudget:
    segment_hours: float
    budget_usd: float
    maximum_rate_usd_per_hour: float
    finalization_reserve_seconds: float = 1800
    transfer_timeout_seconds: float = 600

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError(
                "Segment budget, rate, duration and margins must be finite and positive"
            )
        if self.segment_hours > 11.5:
            raise ValueError("Segments must be <=11.5 hours, leaving room within a 12-hour job cap")
        if self.finalization_reserve_seconds < self.transfer_timeout_seconds + 120:
            raise ValueError(
                "Reserve must cover checkpoint transfer plus validation/serialization/exit"
            )
        if self.finalization_reserve_seconds >= self.seconds:
            raise ValueError("Finalization reserve leaves no training time")
        if self.maximum_platform_cost_usd > self.budget_usd:
            raise ValueError(
                "Requested segment and platform margin exceed the allocated spending ceiling"
            )

    @property
    def seconds(self):
        return self.segment_hours * 3600

    @property
    def platform_max_minutes(self):
        return math.ceil(self.segment_hours * 60) + 5

    @property
    def maximum_platform_cost_usd(self):
        return self.platform_max_minutes / 60 * self.maximum_rate_usd_per_hour

    def report(self):
        return asdict(self) | {
            "platform_max_duration_minutes": self.platform_max_minutes,
            "platform_max_restarts": 0,
            "platform_resume": "none",
            "maximum_platform_compute_cost_usd": self.maximum_platform_cost_usd,
            "account_balance_checked_by_runner": False,
            "separate_build_and_storage_charges_included": False,
            "finalization_reserve_includes": [
                "full validation at Timer stop",
                "checkpoint serialization",
                "upload and complete SHA256 readback",
                "process teardown",
            ],
        }


def verify_streaming_report(path: Path):
    report = json.loads(path.read_text())
    ranks = report.get("ranks", [])
    if (
        report.get("ok") is not True
        or report.get("data_worker_start_method") != "spawn"
        or report.get("workers_per_rank") != 1
        or report.get("cache_refills_observed_on_all_ranks") is not True
        or report.get("checkpoint", {}).get("strict_tensor_and_metadata_roundtrip") is not True
        or len(ranks) != 4
        or any(
            row.get("rank") != rank
            or row.get("world_size") != 4
            or row.get("online_weights_changed") is not True
            or row.get("all_model_parameters_finite") is not True
            for rank, row in enumerate(ranks)
        )
    ):
        raise ValueError(
            "Production requires a successful four-GPU streaming probe with one spawn worker/rank"
        )
    return file_sha256(path)


def freeze_complete_index(source: Path, destination: Path, expected_sha256: str, endpoint: str):
    from src.data.sharded_audio_datamodule import _validate_index

    if not SHA.fullmatch(expected_sha256):
        raise ValueError("A pinned completed index SHA256 is required")
    with source.open("rb") as handle:
        payload = handle.read(16 * 1024**2 + 1)
    if len(payload) > 16 * 1024**2 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen staging index size/SHA256 mismatch")
    value = json.loads(payload)
    _validate_index(value, endpoint, allow_partial=False)
    destination.write_bytes(payload)
    return expected_sha256


def verify_resume_state(path: Path, receipt: dict, index_sha256: str):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    step = receipt["checkpoint"]["step"]
    if (
        state.get("global_step") != step
        or int(state.get("state_dict", {}).get("ema_updates", -1)) != step
        or len(state.get("optimizer_states", [])) != 1
        or not state["optimizer_states"][0].get("state")
        or len(state.get("lr_schedulers", [])) != 1
        or state["lr_schedulers"][0].get("last_epoch") != step
        or any(
            value.get("step") != step for value in state["optimizer_states"][0]["state"].values()
        )
        or not state.get("loops")
        or state.get("ShardedAudioDataModule", {}).get("index_sha256") != index_sha256
    ):
        raise ValueError(
            "Resume checkpoint is not complete matching model/EMA/optimizer/loop/data state"
        )
    from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule

    metadata = HierarchicalWavJEPAModule.validated_checkpoint_metadata(state)
    dynamics = metadata["dynamics"]
    if (
        dynamics["lr_total_steps"],
        dynamics["lr_warmup_steps"],
        dynamics["ema_anneal_end_step"],
    ) != (375000, 100000, 100000) or state.get(
        HierarchicalWavJEPAModule.SCHEDULE_CHECKPOINT_KEY
    ) != {
        "total_steps": 375000,
        "warmup_steps": 100000,
        "ema_end_step": 100000,
    }:
        raise ValueError("Resume checkpoint has a different training schedule")
    return step


def make_config(plan: dict):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train", overrides=["experiment=audioset_h100_four", "data=sharded_audio"]
        )
    OmegaConf.set_struct(cfg, False)
    work, output = Path(plan["work_dir"]), Path(plan["output"])
    cfg.paths.root_dir = cfg.paths.work_dir = str(ROOT)
    cfg.paths.output_dir = str(work / "run")
    cfg.paths.log_dir = str(work / "logs")
    cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
    cfg.logger = None
    cfg.callbacks = {
        "timer": {
            "_target_": "scripts.segment_checkpoint.SegmentTimer",
            "training_deadline": plan["training_deadline"],
        },
        "model_checkpoint": {
            "_target_": "scripts.segment_checkpoint.DurableStepCheckpoint",
            "repository_options": plan["repository_options"],
            "segment_id": plan["segment_id"],
            "data_seed": plan["resolved_data_seed"],
            "receipt_path": str(output / "latest-checkpoint.json"),
            "transfer_timeout_seconds": plan["budget"]["transfer_timeout_seconds"],
            "hard_deadline": plan["hard_deadline"],
            "dirpath": str(work / "run" / "checkpoints"),
            "filename": "step_{step:07d}",
            "auto_insert_metric_name": False,
            "every_n_train_steps": plan["checkpoint_every_steps"],
        },
    }
    cfg.data.index_path = str(work / "frozen-index.json")
    cfg.data.endpoint = plan["data_endpoint"]
    cfg.data.cache_dir = str(work / "shard-cache")
    cfg.data.cache_max_gib = 60
    cfg.data.num_workers = 1
    cfg.data.allow_partial_index = False
    cfg.data.allow_stream_restart_on_resume = True
    cfg.data.seed = plan["resolved_data_seed"]
    cfg.trainer.min_epochs = cfg.trainer.min_steps = 0
    cfg.trainer.max_epochs = -1
    cfg.trainer.max_steps = 375000
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.enable_model_summary = False
    cfg.ckpt_path = plan["resume_path"]
    cfg.train, cfg.test = True, False
    if (
        cfg.data.batch_size * cfg.trainer.devices != 64
        or cfg.trainer.get("accumulate_grad_batches", 1) != 1
    ):
        raise ValueError("Segment must preserve the global optimizer batch of 64")
    if (cfg.model.lr_total_steps, cfg.model.lr_warmup_steps, cfg.model.ema_anneal_end_step) != (
        375000,
        100000,
        100000,
    ):
        raise ValueError("Segment must preserve the complete original LR/EMA schedules")
    return cfg


def worker(plan_path: Path):
    import torch.distributed as dist

    from src.train import train

    plan = json.loads(plan_path.read_text())
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("Training segment requires exactly four externally launched ranks")
    _, objects = train(make_config(plan))
    trainer, model = objects["trainer"], objects["model"]
    step = trainer.global_step
    if step <= plan["resume_step"] or int(model.ema_updates) != step:
        raise RuntimeError("Segment did not advance optimizer and EMA steps together")
    callback = trainer.checkpoint_callback
    if callback.last_receipt is None or callback.last_receipt["checkpoint"]["step"] != step:
        raise RuntimeError("Final optimizer step lacks a verified durable checkpoint")
    dist.barrier()
    if trainer.is_global_zero:
        atomic_json(
            Path(plan["output"]) / "segment-finished.json",
            {
                "segment_finished": True,
                "training_complete": step == 375000,
                "initial_optimizer_step": plan["resume_step"],
                "final_optimizer_step": step,
                "ema_updates": int(model.ema_updates),
                "global_batch": 64,
                "stream_positions_checkpointed": False,
                "stream_restarted_on_resume": plan["resume_path"] is not None,
                "data_seed_policy": "base_plus_resume_step",
                "resolved_data_seed": plan["resolved_data_seed"],
                "checkpoint": callback.last_receipt,
                "budget": plan["budget"],
            },
        )
    dist.barrier()


def remember_descendants(owned):
    """Retain PID/create-time identities even if a worker later becomes orphaned."""
    import psutil

    for process in list(owned.values()):
        try:
            for child in process.children(recursive=True):
                owned[(child.pid, child.create_time())] = child
        except psutil.NoSuchProcess:
            continue


def stop_process_tree(process, owned, *, grace_seconds=10):
    """Stop detached torchrun rank sessions as well as the launcher process.

    Freeze discovered parents before the final tree scan so they cannot create
    more workers during cleanup. psutil checks create times before signaling,
    protecting against PID reuse. Escalation does not depend on wrapper survival.
    """
    import psutil

    suspended = set()
    while True:
        for identity, child in list(owned.items()):
            if identity in suspended:
                continue
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
            suspended.add(identity)
        remember_descendants(owned)
        if set(owned) <= suspended:
            break
    for child in owned.values():
        try:
            child.terminate()
            child.resume()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=grace_seconds)
    # A reparented zombie has released compute resources and awaits init's reap.
    surviving = []
    for child in alive:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                surviving.append(child)
        except psutil.NoSuchProcess:
            pass
    if surviving:
        raise RuntimeError("Owned training processes survived the hard watchdog")
    process.wait(timeout=1)


def segment_budget(args):
    return SegmentBudget(
        args.segment_hours,
        args.budget_usd,
        args.maximum_rate_usd_per_hour,
        args.finalization_reserve_seconds,
        args.transfer_timeout_seconds,
    )


def supervise(args, raw_args):
    """Bound setup and training, including torchrun's detached rank process trees."""
    import psutil

    budget = segment_budget(args)
    if args.plan_only:
        print(json.dumps(budget.report(), indent=2))
        return
    started = time.time()
    hard_stop = time.monotonic() + budget.seconds - 30
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *raw_args,
            "--execute-started",
            str(started),
        ],
        cwd=ROOT,
        start_new_session=True,
    )
    root = psutil.Process(process.pid)
    owned = {(root.pid, root.create_time()): root}
    try:
        while True:
            remember_descendants(owned)
            code = process.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f"Training segment exited unsuccessfully (code {code})")
                break
            # The remaining 30s let TERM/KILL finish within our own bound. The
            # platform's five-minute margin is an additional hard backstop.
            remaining = hard_stop - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, budget.seconds - 30)
            try:
                process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop_process_tree(process, owned)


def execute_segment(args):
    budget = segment_budget(args)
    if not args.exclusive_checkpoint_writer or not args.acknowledge_stream_restart:
        raise ValueError(
            "Explicit exclusive-writer and stream-restart acknowledgements are required"
        )
    if not args.streaming_smoke_report:
        raise ValueError("A successful streaming hardware report is required before launch")
    smoke_sha = verify_streaming_report(args.streaming_smoke_report)
    started = args.execute_started
    hard_deadline = started + budget.seconds
    training_deadline = hard_deadline - budget.finalization_reserve_seconds

    def check():
        if time.time() >= training_deadline:
            raise TimeoutError("Segment setup exhausted its training allowance")

    args.output.mkdir(parents=True, exist_ok=False)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    frozen_sha = freeze_complete_index(
        args.index, args.work_dir / "frozen-index.json", args.index_sha256, args.data_endpoint
    )
    repository_options = dict(
        bucket=args.checkpoint_bucket,
        prefix=args.checkpoint_prefix,
        run_id=args.run_id,
        index_sha256=frozen_sha,
        base_data_seed=args.base_data_seed,
    )
    repository = CheckpointRepository(
        checkpoint_client(args.checkpoint_endpoint), **repository_options, check=check
    )
    repository.open(initialize=not args.resume)
    resume_path, resume_step = None, 0
    if args.resume:
        resume_path = args.work_dir / "resume.ckpt"
        receipt = repository.download_latest(resume_path)
        resume_step = verify_resume_state(resume_path, receipt, frozen_sha)
        atomic_json(args.output / "resumed-checkpoint.json", receipt)
        if resume_step == 375000:
            raise ValueError("Training already completed 375000 steps; no new segment is needed")
    check()
    plan = {
        "segment_id": args.segment_id,
        "work_dir": str(args.work_dir.resolve()),
        "output": str(args.output.resolve()),
        "repository_options": repository_options | {"endpoint": args.checkpoint_endpoint},
        "data_endpoint": args.data_endpoint,
        "index_sha256": frozen_sha,
        "streaming_smoke_report_sha256": smoke_sha,
        "training_deadline": training_deadline,
        "hard_deadline": hard_deadline,
        "budget": budget.report(),
        "resume_path": str(resume_path.resolve()) if resume_path else None,
        "resume_step": resume_step,
        "data_seed_policy": "base_plus_resume_step",
        "base_data_seed": args.base_data_seed,
        "resolved_data_seed": args.base_data_seed + resume_step,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "stream_positions_checkpointed": False,
    }
    plan_path = args.output / "segment-plan.json"
    atomic_json(plan_path, plan)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nproc_per_node=4",
        str(Path(__file__).resolve()),
        "--worker-plan",
        str(plan_path.resolve()),
    ]
    # torchrun creates a new session for each rank. The parent's watchdog tracks
    # descendant PID/create-time identities rather than assuming shared PGIDs.
    code = subprocess.Popen(command, cwd=ROOT, start_new_session=False).wait()
    if code:
        raise RuntimeError(f"Training segment exited unsuccessfully (code {code})")
    result_path = args.output / "segment-finished.json"
    result = json.loads(result_path.read_text())
    if (
        result.get("segment_finished") is not True
        or result.get("final_optimizer_step", 0) <= resume_step
    ):
        raise RuntimeError("Worker exited without an advanced, verified final checkpoint")
    result["wall_seconds"] = time.time() - started
    result["observed_runner_time_cost_at_supplied_rate_usd"] = (
        result["wall_seconds"] / 3600 * budget.maximum_rate_usd_per_hour
    )
    result["compute_cost_upper_bound_usd"] = budget.maximum_platform_cost_usd
    atomic_json(args.output / "segment-success.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(Path(os.environ["GMN_RESULT_PATH"]), result)
    print(
        json.dumps(
            {
                "segment_finished": True,
                "final_optimizer_step": result["final_optimizer_step"],
                "training_complete": result["training_complete"],
            }
        ),
        flush=True,
    )


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--execute-started", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--segment-hours", type=float, default=10)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--maximum-rate-usd-per-hour", type=float)
    parser.add_argument("--finalization-reserve-seconds", type=float, default=1800)
    parser.add_argument("--transfer-timeout-seconds", type=float, default=600)
    parser.add_argument("--run-id")
    parser.add_argument("--segment-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-sha256")
    parser.add_argument("--data-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--checkpoint-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--checkpoint-bucket")
    parser.add_argument("--checkpoint-prefix")
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument("--base-data-seed", type=int, default=42)
    parser.add_argument("--streaming-smoke-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclusive-checkpoint-writer", action="store_true")
    parser.add_argument("--acknowledge-stream-restart", action="store_true")
    args = parser.parse_args(raw_args)
    if args.worker_plan:
        return worker(args.worker_plan)
    required = ["budget_usd", "maximum_rate_usd_per_hour"]
    if not args.plan_only:
        required += [
            "run_id",
            "segment_id",
            "output",
            "work_dir",
            "index",
            "index_sha256",
            "checkpoint_bucket",
            "checkpoint_prefix",
        ]
    if any(getattr(args, field) is None for field in required):
        parser.error(
            "Missing required launch fields: "
            + ", ".join(field for field in required if getattr(args, field) is None)
        )
    if args.checkpoint_every_steps < 1:
        parser.error("checkpoint-every-steps must be positive")
    if args.execute_started is not None:
        execute_segment(args)
    else:
        supervise(args, raw_args)


if __name__ == "__main__":
    main()

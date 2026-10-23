"""Bounded real-audio throughput comparison; never starts cloud resources.

The supervisor downloads a verified small AudioSet pilot once and launches
isolated DDP runs. End-to-end timing includes batch loading after warmup, but
the pilot stays on local disk: full-corpus remote cache misses remain separate.
Keep platform runtime/build caps and max_restarts=0 around this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

if __name__ == "__main__":
    print(json.dumps({"phase": "import_training_libraries"}), flush=True)

import lightning as L
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PILOT_SHA = "645f2327738ce0e0fec6a192bd05f2f8e806003e2865b64030037299ac646572"
PILOT_BYTES = 771368960
VARIANTS = {
    "dense_checkpointed_2": (2, True, False),
    "dense_2": (2, False, False),
    "compact_2": (2, False, True),
    "dense_4": (4, False, False),
    "compact_4": (4, False, True),
}


def atomic_report(path, report):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_pilot(destination: Path) -> None:
    local_archive = os.environ.get("HWAVJEPA_LOCAL_PILOT")
    phase = "verify_local_pilot" if local_archive else "download_verified_pilot"
    print(json.dumps({"phase": phase, "bytes": PILOT_BYTES}), flush=True)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if local_archive:
            archive = Path(local_archive).expanduser().resolve(strict=True)
            if not archive.is_file() or archive.stat().st_size != PILOT_BYTES:
                raise ValueError("Pilot size or SHA256 mismatch")
            with archive.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != PILOT_SHA:
                raise ValueError("Pilot size or SHA256 mismatch")
        else:
            from scripts.probe_sharded_audio import deadline
            from src.stage_audioset import make_client

            archive = destination / "pilot.tar"
            digest, count = hashlib.sha256(), 0
            # One wall-clock bound covers connection/request retries and body reads.
            # The cancellation is BaseException-derived so SDK retries cannot swallow it.
            with deadline(120):
                client = make_client(os.environ["HWAVJEPA_S3_ENDPOINT"])
                result = client.get_object(
                    Bucket=os.environ["HWAVJEPA_PILOT_BUCKET"],
                    Key=os.environ["HWAVJEPA_PILOT_KEY"],
                )
                try:
                    with archive.open("wb") as handle:
                        for chunk in result["Body"].iter_chunks(1024**2):
                            count += len(chunk)
                            if count > PILOT_BYTES:
                                raise ValueError("Pilot exceeds declared size")
                            handle.write(chunk)
                            digest.update(chunk)
                finally:
                    result["Body"].close()
            if count != PILOT_BYTES or digest.hexdigest() != PILOT_SHA:
                raise ValueError("Pilot size or SHA256 mismatch")
        with tarfile.open(archive) as source:
            for member in source:
                path = destination / member.name
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Pilot archive contains an unsafe path")
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as payload, path.open("wb") as output:
                        shutil.copyfileobj(payload, output)
                else:
                    raise ValueError("Pilot archive contains a non-regular member")
        if not local_archive:
            archive.unlink()
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            if not (destination / name).is_file():
                raise ValueError(f"Pilot has no {name}")
    except BaseException:
        # mkdir(exist_ok=False) above proves ownership. Never delete the image archive.
        shutil.rmtree(destination, ignore_errors=True)
        raise
    print(json.dumps({"phase": "pilot_ready", "sha256": PILOT_SHA}), flush=True)


class SustainedTimings(L.Callback):
    def __init__(self, output: str, steps: int, warmup: int, variant: str):
        self.output, self.steps, self.warmup = Path(output), steps, warmup
        self.variant = variant
        self.ends = []
        self.started = None

    def on_train_start(self, trainer, pl_module):
        self.initial = next(pl_module.backbone.model.encoder.parameters()).detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if trainer.global_step == self.warmup:
            self.started = now
            torch.cuda.reset_peak_memory_stats()
        elif trainer.global_step > self.warmup:
            self.ends.append(now)

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        if trainer.global_step != self.steps or int(pl_module.ema_updates) != self.steps:
            raise RuntimeError("Optimizer and EMA step counts differ from the requested benchmark")
        if self.started is None or len(self.ends) != self.steps - self.warmup:
            raise RuntimeError("Incomplete measured throughput window")
        intervals = [
            end - start
            for start, end in zip([self.started, *self.ends[:-1]], self.ends, strict=True)
        ]
        changed = not torch.equal(
            self.initial, next(pl_module.backbone.model.encoder.parameters()).detach()
        )
        if not changed:
            raise RuntimeError("Online encoder weights did not change")
        report = {
            "variant": self.variant,
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
            "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            "online_weights_changed": changed,
            "input": "verified AudioSet pilot on local disk; real WAV reads and epoch crops",
            "pilot_sha256": PILOT_SHA,
        }
        atomic_report(self.output / f"rank-{trainer.global_rank}.json", report)


def worker(args):
    from src.train import train

    devices, checkpointing, compact = VARIANTS[args.variant]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["experiment=audioset_h100_fast"])
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir, cfg.paths.work_dir = str(ROOT), str(ROOT)
    cfg.paths.output_dir = str(args.output / "run")
    cfg.paths.log_dir = str(args.output / "logs")
    cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
    cfg.logger = None
    cfg.callbacks = {
        "timing": {
            "_target_": "scripts.benchmark_gpu.SustainedTimings",
            "output": str(args.output),
            "steps": args.steps,
            "warmup": args.warmup,
            "variant": args.variant,
        }
    }
    for split in ("train", "val", "test"):
        cfg.data[f"{split}_manifest"] = str(args.data / f"{split}.jsonl")
    cfg.data.batch_size = 64 // devices
    cfg.data.num_workers = 2
    cfg.trainer.devices = devices
    cfg.trainer.max_steps = args.steps
    cfg.trainer.limit_val_batches = 0
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    # Keep the real 375k LR/EMA schedules; do not substitute a short probe schedule.
    cfg.model.backbone.config.use_gradient_checkpointing = checkpointing
    cfg.model.backbone.config.compact_masked_tokens = compact
    _, objects = train(cfg)
    trainer = objects["trainer"]
    dist.barrier()
    checkpoint_started = time.perf_counter()
    checkpoint = args.output.parent / f"{args.variant}-temporary.ckpt"
    trainer.save_checkpoint(checkpoint, weights_only=False)
    dist.barrier()
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    if trainer.is_global_zero:
        atomic_report(
            args.output / "checkpoint.json",
            {"seconds": checkpoint_seconds, "bytes": checkpoint.stat().st_size},
        )
        checkpoint.unlink()
    dist.barrier()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Supervisor requires four visible BF16-capable CUDA GPUs")
    prepare_pilot(args.data)
    hardware = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(i) for i in range(4)],
    }
    atomic_report(args.output / "hardware.json", hardware)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_localereg_performance.py",
            "-k",
            "cuda",
            "-q",
            f"--junitxml={args.output / 'cuda-parity.xml'}",
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
    )
    cases = ET.parse(args.output / "cuda-parity.xml").findall(".//testcase")
    if len(cases) != 1 or cases[0].find("skipped") is not None:
        raise RuntimeError("The CUDA regularizer parity check must execute")
    results = []
    for variant in args.variants:
        print(json.dumps({"phase": "benchmark", "variant": variant}), flush=True)
        devices = VARIANTS[variant][0]
        output = args.output / variant
        output.mkdir()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(devices)))}
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            f"--nproc_per_node={devices}",
            str(Path(__file__).resolve()),
            "--worker",
            "--variant",
            variant,
            "--output",
            str(output),
            "--data",
            str(args.data),
            "--steps",
            str(args.steps),
            "--warmup",
            str(args.warmup),
        ]
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        timed_out = False
        try:
            returncode = process.wait(timeout=360)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
        entry = {
            "variant": variant,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_seconds": time.perf_counter() - started,
        }
        if returncode == 0:
            ranks = [json.loads((output / f"rank-{i}.json").read_text()) for i in range(devices)]
            seconds = max(row["mean_seconds_per_update"] for row in ranks)
            entry.update(
                {
                    "ranks": ranks,
                    "seconds_per_update": seconds,
                    "updates_per_second": 1 / seconds,
                    "projected_375k_train_hours": seconds * 375000 / 3600,
                    "standard_batch_gpu_cost_usd": seconds * 375000 / 60 * devices * 0.063,
                    "checkpoint": json.loads((output / "checkpoint.json").read_text()),
                }
            )
        results.append(entry)
        atomic_report(args.output / "results.json", {"hardware": hardware, "variants": results})
        print(json.dumps({k: v for k, v in entry.items() if k != "ranks"}), flush=True)
    successful = [row for row in results if row["returncode"] == 0]
    if not successful:
        raise RuntimeError("Every benchmark configuration failed")
    summary = {
        "ok": True,
        "hardware": hardware,
        "variants": results,
        "limitations": [
            "Small real AudioSet pilot reused across epochs; remote cache misses excluded.",
            "Validation, checkpoint upload and queue time are outside steady-state timing.",
            "This measures speed and finite updates, not convergence or HEAR quality.",
        ],
    }
    atomic_report(args.output / "benchmark-success.json", summary)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_report(Path(os.environ["GMN_RESULT_PATH"]), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("/tmp/h-wavjepa-benchmark-audio"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=225)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 1 or (args.worker and not args.variant):
        parser.error("Require steps > warmup >= 1 and a variant for each worker")
    worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()

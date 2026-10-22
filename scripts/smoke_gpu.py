"""Bounded two-GPU correctness and full-base capacity probe; no dataset download.

Run inside the prepared GPU image: python scripts/smoke_gpu.py --output /output
The full model uses generated waveforms at the AudioSet preset's exact shape and
batch size. This measures implementation/capacity, not AudioSet learning quality.
An external job runtime/build timeout is required to bound cloud charges.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StepTimings(L.Callback):
    def __init__(self, output: str) -> None:
        self.output = Path(output)
        self.rows = []

    def on_train_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        self.rows.append(
            {
                "step": trainer.global_step,
                "seconds": time.perf_counter() - self.started,
                "allocated_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )

    def on_train_end(self, trainer, pl_module):
        assert trainer.global_step == pl_module.ema_updates.item() == 3
        report = {
            "rank": trainer.global_rank,
            "steps": trainer.global_step,
            "ema_updates": int(pl_module.ema_updates),
            "batch_per_gpu": 32,
            "global_batch": 64,
            "samples_per_recording": 145440,
            "input": "generated waveforms, AudioSet preset shape; no downloaded AudioSet",
            "timings": self.rows,
        }
        (self.output / f"base-rank-{trainer.global_rank}.json").write_text(
            json.dumps(report, indent=2)
        )


def full_base_worker(output: Path) -> None:
    from src.train import train

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["experiment=audioset_h100"])
    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir = str(ROOT)
    cfg.paths.output_dir = str(output / "base-run")
    cfg.paths.work_dir = str(ROOT)
    cfg.paths.log_dir = str(output / "logs")
    cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
    cfg.logger = None
    cfg.callbacks = {"timing": {"_target_": "scripts.smoke_gpu.StepTimings", "output": str(output)}}
    cfg.data = {
        "_target_": "src.data.synthetic_datamodule.SyntheticAudioDataModule",
        "sample_rate": 16000,
        "num_samples": 145440,
        "batch_size": 32,
        "train_size": 256,
        "val_size": 64,
        "test_size": 64,
        "num_workers": 0,
        "seed": 42,
    }
    # SyntheticAudioDataModule uses Lightning's sampler; the preceding correctness
    # test exercises WAVManifestDataModule's own sampler and checkpoint/resume.
    cfg.trainer.use_distributed_sampler = True
    cfg.trainer.max_steps = 3
    cfg.trainer.limit_train_batches = 3
    cfg.trainer.limit_val_batches = 0
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.enable_checkpointing = False
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    cfg.trainer.log_every_n_steps = 1
    cfg.trainer.val_check_interval = 1.0
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.model.lr_total_steps = 3
    cfg.model.lr_warmup_steps = 0
    cfg.model.ema_anneal_end_step = 3
    train(cfg)


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 2 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This probe requires exactly two visible BF16-capable CUDA GPUs")
    hardware = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "memory_gib": torch.cuda.get_device_properties(index).total_memory / 2**30,
            }
            for index in range(2)
        ],
    }
    (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_dynamics_ddp.py",
            "-k",
            "cuda",
            f"--junitxml={output / 'ddp-correctness.xml'}",
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
    cases = ET.parse(output / "ddp-correctness.xml").findall(".//testcase")
    if len(cases) != 1 or cases[0].find("skipped") is not None:
        raise RuntimeError("The CUDA DDP test must execute, not skip")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        timeout=360,
    )
    reports = [json.loads((output / f"base-rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(report["steps"] == report["ema_updates"] == 3 for report in reports)
    result = {
        "ok": True,
        "cuda_ddp_correctness": True,
        "full_base_batch_32": True,
        "reports": reports,
    }
    (output / "smoke-success.json").write_text(json.dumps(result, indent=2))
    if os.environ.get("GMN_RESULT_PATH"):
        Path(os.environ["GMN_RESULT_PATH"]).write_text(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    full_base_worker(args.output) if args.worker else run(args.output)

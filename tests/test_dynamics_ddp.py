"""Actual two-process Lightning training, checkpointing and resume on WAV data."""

import json
import os
import shutil
import signal
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def ddp_wav_dataset(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ddp-ten-second-wavs")
    # 24 clips use under 8 MB. Every recording contains nonconstant chronological
    # content so random-offset reading, PCM decoding and real sharding are exercised.
    samples = np.arange(160_000, dtype=np.float64)
    for split, count in (("train", 16), ("val", 4), ("test", 4)):
        rows = []
        for index in range(count):
            path = folder / f"{split}-{index}.wav"
            waveform = (10_000 * np.sin(samples * (0.017 + index * 0.001))).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                handle.writeframes(waveform.tobytes())
            rows.append(json.dumps({"path": path.name, "recording_id": path.stem}))
        (folder / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    try:
        yield folder
    finally:
        shutil.rmtree(folder)


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo backend")
@pytest.mark.parametrize(
    ("accelerator", "precision"),
    [
        ("cpu", "32-true"),
        ("cpu", "bf16-mixed"),
        pytest.param(
            "cuda",
            "bf16-mixed",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2 or not torch.cuda.is_bf16_supported(),
                reason="Requires two CUDA devices with BF16 support",
            ),
        ),
    ],
)
def test_two_process_wav_training_checkpoint_resume_and_ema(
    ddp_wav_dataset, tmp_path, accelerator, precision
):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    # The subprocess owns its ranks/launcher. Do not inherit an enclosing launcher.
    for name in (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(root),
        TOKENIZERS_PARALLELISM="false",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            "--nnodes=1",
            "--nproc_per_node=2",
            str(root / "tests" / "ddp_dynamics_worker.py"),
            str(root),
            str(ddp_wav_dataset),
            str(tmp_path),
            accelerator,
            precision,
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        # Stop the worker ranks as well as torchrun, so a failed collective does
        # not leave orphan trainers using GPUs after the test's timeout.
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        pytest.fail(f"Two-process training timed out:\n{output}")
    assert process.returncode == 0, output
    for phase, expected_steps in (("initial", 2), ("resumed", 4)):
        output = tmp_path / phase
        reports = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
        assert reports[0]["state_sha256"] == reports[1]["state_sha256"]
        assert all(report["global_step"] == expected_steps for report in reports)
        assert all(report["ema_updates"] == expected_steps for report in reports)
        indices = [{index[0] for index in report["sample_indices"]} for report in reports]
        assert indices[0].isdisjoint(indices[1])
        assert indices[0] | indices[1] == set(range(16))
        assert (output / "checkpoints" / "last.ckpt").is_file()
        checkpoint = torch.load(
            output / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=True
        )
        assert checkpoint["global_step"] == expected_steps
        assert checkpoint["state_dict"]["ema_updates"].item() == expected_steps
        run = json.loads((output / "run.json").read_text())
        assert run["status"] == "completed"
        assert run["global_step"] == expected_steps
    initial = json.loads((tmp_path / "initial" / "rank-0.json").read_text())
    resumed = json.loads((tmp_path / "resumed" / "rank-0.json").read_text())
    assert initial["state_sha256"] != resumed["state_sha256"]
    assert resumed["sampler_epoch"] > initial["sampler_epoch"]

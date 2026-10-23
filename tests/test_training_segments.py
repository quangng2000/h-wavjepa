"""Bounded segment configuration, verified S3 handoffs and real Lightning resume."""

import io
import json
import subprocess
import sys
import time
from copy import deepcopy
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from test_stage_audioset import FakeS3
from torch.utils.data import DataLoader, IterableDataset

from scripts.segment_checkpoint import (
    CheckpointRepository,
    DurableStepCheckpoint,
    SegmentTimer,
    checkpoint_client,
)
from scripts.train_segment import (
    SegmentBudget,
    main,
    make_config,
    remember_descendants,
    stop_process_tree,
    supervise,
    verify_resume_state,
    verify_streaming_report,
)
from src.utils.runtime import validate_config


class CheckpointS3(FakeS3):
    def __init__(self):
        super().__init__()
        self.deleted = []
        self.events = []
        self.corrupt_checkpoint = False
        self.fail_latest = False
        self.fail_checkpoint = False

    def delete_object(self, *, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)

    def get_object(self, **kwargs):
        self.events.append(("get", kwargs["Key"]))
        result = super().get_object(**kwargs)
        if self.corrupt_checkpoint and kwargs["Key"].endswith(".ckpt"):
            value = result["Body"].read()
            result["Body"] = io.BytesIO(b"x" + value[1:])
        return result

    def put_object(self, **kwargs):
        self.events.append(("put", kwargs["Key"]))
        if self.fail_latest and kwargs["Key"].endswith("latest.json"):
            raise OSError("simulated receipt upload failure")
        if self.fail_checkpoint and kwargs["Key"].endswith(".ckpt"):
            raise OSError("simulated checkpoint upload failure")
        return super().put_object(**kwargs)


OPTIONS = {
    "bucket": "checkpoint-only",
    "prefix": "training/training-run-001",
    "run_id": "training-run-001",
    "index_sha256": "a" * 64,
}


def repository(client, **changes):
    return CheckpointRepository(client, **(OPTIONS | changes), check=lambda: None)


def payload(tmp_path, step):
    path = tmp_path / f"state-{step}.ckpt"
    path.write_bytes(f"complete-state-at-step-{step}".encode())
    return path


def test_two_owned_checkpoints_retained_only_after_full_remote_readback(tmp_path):
    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    receipts = [
        repo.publish(payload(tmp_path, step), step, segment_id="segment-001", data_seed=42)
        for step in (2, 4, 6)
    ]
    assert receipts[-1]["previous"] == receipts[-2]["checkpoint"]
    assert client.deleted == [receipts[0]["checkpoint"]["key"]]
    checkpoint_keys = [key for key in client.objects if key.endswith(".ckpt")]
    assert len(checkpoint_keys) == 2
    final_key = receipts[-1]["checkpoint"]["key"]
    assert client.events.index(("get", final_key)) < max(
        index for index, item in enumerate(client.events) if item == ("put", repo.latest_key)
    )
    restored = tmp_path / "restored.ckpt"
    assert repo.download_latest(restored) == receipts[-1]
    assert restored.read_bytes() == payload(tmp_path, 6).read_bytes()
    with pytest.raises(ValueError, match="explicit resume"):
        repo.open(initialize=True)
    assert repo.open(initialize=False) == receipts[-1]


@pytest.mark.parametrize("failure", ["readback", "latest"])
def test_failed_publication_keeps_previous_durable_checkpoint(tmp_path, failure):
    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    previous = repo.publish(payload(tmp_path, 2), 2, segment_id="segment-001", data_seed=42)
    prior_receipt = client.objects[repo.latest_key]
    client.corrupt_checkpoint = failure == "readback"
    client.fail_latest = failure == "latest"
    with pytest.raises((ValueError, OSError)):
        repo.publish(payload(tmp_path, 4), 4, segment_id="segment-002", data_seed=44)
    assert client.objects[repo.latest_key] == prior_receipt
    assert previous["checkpoint"]["key"] in client.objects
    assert client.deleted == []
    assert len([key for key in client.objects if key.endswith(".ckpt")]) <= 2
    client.corrupt_checkpoint = client.fail_latest = False
    repo.download_latest(tmp_path / "recovered.ckpt")
    assert (tmp_path / "recovered.ckpt").read_bytes() == payload(tmp_path, 2).read_bytes()


def test_resume_rejects_corrupt_bytes_even_when_size_and_receipt_match(tmp_path):
    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    repo.publish(payload(tmp_path, 2), 2, segment_id="segment-001", data_seed=42)
    client.corrupt_checkpoint = True
    with pytest.raises(ValueError, match="SHA256"):
        repo.download_latest(tmp_path / "resume.ckpt")
    assert not (tmp_path / "resume.ckpt").exists()
    assert not (tmp_path / "resume.partial").exists()


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_damaged_latest_never_causes_pruning_last_recoverable_fallback(tmp_path, damage):
    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    first, second = [
        repo.publish(payload(tmp_path, step), step, segment_id="segment-001", data_seed=42)
        for step in (2, 4)
    ]
    latest_key = second["checkpoint"]["key"]
    if damage == "missing":
        del client.objects[latest_key]
    else:
        client.objects[latest_key] = b"x" + client.objects[latest_key][1:]
    client.fail_checkpoint = True
    with pytest.raises(ValueError, match="missing|checksum"):
        repo.publish(payload(tmp_path, 6), 6, segment_id="segment-002", data_seed=46)
    assert first["checkpoint"]["key"] in client.objects
    assert client.deleted == []


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "training-run-001",
        "/training/training-run-001",
        "training/../training-run-001",
        "training/other-run",
    ],
)
def test_cleanup_requires_explicit_owned_run_prefix(prefix):
    with pytest.raises(ValueError, match="Owned prefix"):
        repository(CheckpointS3(), prefix=prefix)


def test_foreign_or_changed_prefix_is_never_adopted_or_deleted(tmp_path):
    client = CheckpointS3()
    repo = repository(client)
    foreign = repo.prefix + "/someone-elses-file"
    client.objects[foreign] = b"keep"
    with pytest.raises(ValueError, match="unowned/nonempty"):
        repo.open(initialize=True)
    del client.objects[foreign]
    repo.open(initialize=True)
    client.objects[repo.prefix + "/checkpoints/foreign.ckpt"] = b"keep"
    with pytest.raises(ValueError, match="Foreign object"):
        repo.publish(payload(tmp_path, 2), 2, segment_id="segment-001", data_seed=42)
    assert client.deleted == []
    with pytest.raises(ValueError, match="different run"):
        repository(client, index_sha256="b" * 64).open(initialize=False)


def test_out_of_prefix_receipt_cannot_delete_or_download_another_object(tmp_path):
    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    receipt = repo.publish(payload(tmp_path, 2), 2, segment_id="segment-001", data_seed=42)
    receipt["checkpoint"]["key"] = "other-project/checkpoint.ckpt"
    client.objects[repo.latest_key] = json.dumps(receipt).encode()
    with pytest.raises(ValueError, match="out-of-prefix"):
        repo.download_latest(tmp_path / "resume.ckpt")
    assert client.deleted == []


def test_checkpoint_credentials_are_separate_from_read_only_audio_credentials(monkeypatch):
    import boto3

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "audio-read-only")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "audio-secret-fixture")
    monkeypatch.delenv("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(ValueError, match="separate"):
        checkpoint_client("https://storage.example.invalid")
    monkeypatch.setenv("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID", "checkpoint-writer")
    monkeypatch.setenv("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY", "checkpoint-secret-fixture")
    captured = {}

    def client(service, **options):
        captured.update(options)
        return "fixture-client"

    monkeypatch.setattr(boto3, "client", client)
    assert checkpoint_client("https://storage.example.invalid") == "fixture-client"
    assert captured["aws_access_key_id"] == "checkpoint-writer"
    assert captured["aws_secret_access_key"] == "checkpoint-secret-fixture"
    assert __import__("os").environ["AWS_ACCESS_KEY_ID"] == "audio-read-only"


def test_budget_includes_platform_margin_and_cannot_silently_exceed_allocation(capsys):
    budget = SegmentBudget(10, 160, 15.12)
    assert budget.platform_max_minutes == 605
    assert budget.maximum_platform_cost_usd == pytest.approx(152.46)
    assert budget.report()["platform_max_restarts"] == 0
    with pytest.raises(ValueError, match="spending ceiling"):
        SegmentBudget(10, 150, 15.12)
    with pytest.raises(ValueError, match="11.5"):
        SegmentBudget(12, 500, 15.12)
    with pytest.raises(ValueError, match="Reserve"):
        SegmentBudget(10, 160, 15.12, 500, 600)
    main(["--plan-only", "--budget-usd", "160", "--maximum-rate-usd-per-hour", "15.12"])
    assert json.loads(capsys.readouterr().out)["account_balance_checked_by_runner"] is False


def test_hardware_gate_rejects_fork_or_incomplete_streaming_evidence(tmp_path):
    report = {
        "ok": True,
        "data_worker_start_method": "spawn",
        "workers_per_rank": 1,
        "cache_refills_observed_on_all_ranks": True,
        "checkpoint": {"strict_tensor_and_metadata_roundtrip": True},
        "ranks": [
            {
                "rank": rank,
                "world_size": 4,
                "online_weights_changed": True,
                "all_model_parameters_finite": True,
            }
            for rank in range(4)
        ],
    }
    path = tmp_path / "streaming.json"
    path.write_text(json.dumps(report))
    assert len(verify_streaming_report(path)) == 64
    for key, value in (
        ("data_worker_start_method", "fork"),
        ("ok", False),
        ("workers_per_rank", 2),
    ):
        changed = report | {key: value}
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="streaming probe"):
            verify_streaming_report(path)


def plan(tmp_path):
    return {
        "work_dir": str(tmp_path / "work"),
        "output": str(tmp_path / "output"),
        "training_deadline": time.time() + 3600,
        "hard_deadline": time.time() + 5400,
        "repository_options": OPTIONS | {"endpoint": "https://storage.example.invalid"},
        "segment_id": "segment-001",
        "budget": SegmentBudget(10, 160, 15.12).report(),
        "checkpoint_every_steps": 5000,
        "data_endpoint": "https://storage.givemeanode.com",
        "resume_path": "/tmp/verified-resume.ckpt",
        "resolved_data_seed": 1042,
    }


def test_production_config_preserves_objective_global_batch_and_absolute_schedule(tmp_path):
    cfg = make_config(plan(tmp_path))
    validate_config(cfg)
    assert cfg.trainer.devices == 4 and cfg.data.batch_size == 16
    assert cfg.data.num_workers == 1 and cfg.data.allow_partial_index is False
    assert cfg.data.allow_stream_restart_on_resume is True
    assert cfg.trainer.min_epochs == cfg.trainer.min_steps == 0
    assert cfg.trainer.max_steps == cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.model.readout_mode == "normalized_top_k"
    assert cfg.model.masked_weight == 1 and cfg.model.regularization_weight == 0.1
    assert cfg.trainer.gradient_clip_val == 1 and cfg.model.optimizer.fused is False
    assert cfg.ckpt_path == "/tmp/verified-resume.ckpt"
    assert cfg.seed == 42 and cfg.data.seed == 1042
    assert list(cfg.callbacks) == ["timer", "model_checkpoint"]


def test_segment_data_seed_changes_deterministically_without_model_or_schedule_change(tmp_path):
    initial = make_config(plan(tmp_path) | {"resume_path": None, "resolved_data_seed": 42})
    resumed = make_config(plan(tmp_path) | {"resolved_data_seed": 42 + 50000})
    repeated = make_config(plan(tmp_path) | {"resolved_data_seed": 42 + 50000})
    assert initial.data.seed == 42
    assert resumed.data.seed == repeated.data.seed == 50042
    assert initial.seed == resumed.seed == repeated.seed == 42
    assert initial.model == resumed.model == repeated.model


def assert_resume_counters_rejected(state, receipt, tmp_path):
    from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule

    for field in ("optimizer", "scheduler", "schedule"):
        changed = deepcopy(state)
        if field == "optimizer":
            first = next(iter(changed["optimizer_states"][0]["state"].values()))
            first["step"] -= 1
        elif field == "scheduler":
            changed["lr_schedulers"][0]["last_epoch"] -= 1
        else:
            changed[HierarchicalWavJEPAModule.SCHEDULE_CHECKPOINT_KEY]["warmup_steps"] = 1
        path = tmp_path / f"mismatched-{field}.ckpt"
        torch.save(changed, path)
        with pytest.raises(ValueError, match="matching|schedule"):
            verify_resume_state(path, receipt, "a" * 64)


class InfiniteWaves(IterableDataset):
    def __iter__(self):
        wave = torch.sin(torch.arange(512 * 9) / 13)
        while True:
            yield {"waveform": wave}


class ShardedAudioDataModule(L.LightningDataModule):
    """Tiny local fixture with the production data checkpoint identity."""

    def train_dataloader(self):
        return DataLoader(InfiniteWaves(), batch_size=2, num_workers=0)

    def val_dataloader(self):
        waveform = torch.sin(torch.arange(512 * 9) / 13)
        return DataLoader([{"waveform": waveform}] * 2, batch_size=2)

    def state_dict(self):
        return {"index_sha256": "a" * 64, "stream_position_checkpointed": False}

    def load_state_dict(self, state_dict):
        assert state_dict == self.state_dict()


@pytest.mark.integration
def test_timer_resume_preserves_optimizer_ema_schedule_and_durable_latest(tmp_path, monkeypatch):
    from test_hierarchical_dynamics import model_factory

    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    monkeypatch.setattr("scripts.segment_checkpoint.checkpoint_client", lambda endpoint: client)
    callbacks, trainers = [], []
    saved = None

    class VerifyDurableBeforeValidation(L.Callback):
        def __init__(self):
            self.observed_steps = []

        def on_validation_start(self, trainer, pl_module):
            assert repo.latest()["checkpoint"]["step"] == trainer.global_step
            self.observed_steps.append(trainer.global_step)

    for number, stop in ((1, 2), (2, 4)):
        timer = SegmentTimer(time.time() + 60)
        normal_elapsed = timer.time_elapsed
        callback = DurableStepCheckpoint(
            repository_options=OPTIONS | {"endpoint": "https://storage.example.invalid"},
            segment_id=f"segment-{number}",
            data_seed=42 + (0 if number == 1 else 2),
            receipt_path=str(tmp_path / f"receipt-{number}.json"),
            transfer_timeout_seconds=30,
            hard_deadline=time.time() + 600,
            dirpath=tmp_path / f"local-{number}",
            every_n_train_steps=5000,
        )
        validator = VerifyDurableBeforeValidation()
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=375000,
            max_epochs=-1,
            min_epochs=0,
            min_steps=0,
            callbacks=[timer, callback, validator],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=1,
            check_val_every_n_epoch=None,
            val_check_interval=5000,
            num_sanity_val_steps=0,
            default_root_dir=tmp_path / f"run-{number}",
        )
        monkeypatch.setattr(
            timer,
            "time_elapsed",
            lambda stage="train", t=trainer, n=normal_elapsed, target=stop: (
                120.0 if stage == "train" and t.global_step >= target else n(stage)
            ),
        )
        model = model_factory(
            lr_schedule="cosine",
            lr_total_steps=375000,
            lr_warmup_steps=100000,
            ema_anneal_end_step=100000,
            ema_end_decay=0.99999,
        )
        trainer.fit(
            model,
            datamodule=ShardedAudioDataModule(),
            ckpt_path=str(saved) if saved else None,
            weights_only=True,
        )
        assert trainer.global_step == int(model.ema_updates) == stop
        assert validator.observed_steps == [stop]
        assert timer._offset == 0
        assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == stop
        assert model._resolved_training_schedule == {
            "total_steps": 375000,
            "warmup_steps": 100000,
            "ema_end_step": 100000,
        }
        receipt = repo.latest()
        assert receipt["checkpoint"]["step"] == stop
        assert receipt["resolved_data_seed"] == 42 + (0 if number == 1 else 2)
        saved = tmp_path / f"resume-{number}.ckpt"
        repo.download_latest(saved)
        assert verify_resume_state(saved, receipt, "a" * 64) == stop
        state = torch.load(saved, weights_only=True)
        assert all(
            int(item["step"]) == stop for item in state["optimizer_states"][0]["state"].values()
        )
        assert state["callbacks"][timer.state_key]["time_elapsed"]["train"] == 120.0
        if number == 1:
            assert_resume_counters_rejected(state, receipt, tmp_path)
        callbacks.append(callback)
        trainers.append(trainer)
    assert (
        trainers[1].optimizers[0].param_groups[0]["lr"]
        > trainers[0].optimizers[0].param_groups[0]["lr"]
    )
    assert len([key for key in client.objects if key.endswith(".ckpt")]) == 2


@pytest.mark.integration
def test_failed_upload_stops_lightning_without_success_receipt(tmp_path, monkeypatch):
    from test_step_checkpoint import InfiniteOnes, TinyModel

    client = CheckpointS3()
    repo = repository(client)
    repo.open(initialize=True)
    client.fail_latest = True
    monkeypatch.setattr("scripts.segment_checkpoint.checkpoint_client", lambda endpoint: client)
    callback = DurableStepCheckpoint(
        repository_options=OPTIONS | {"endpoint": "https://storage.example.invalid"},
        segment_id="segment-001",
        data_seed=42,
        receipt_path=str(tmp_path / "receipt.json"),
        transfer_timeout_seconds=30,
        hard_deadline=time.time() + 600,
        dirpath=tmp_path / "local",
        every_n_train_steps=1,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        callbacks=[callback],
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Durable checkpoint publication failed"):
        trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert not (tmp_path / "receipt.json").exists()
    assert repo.latest() is None


def test_watchdog_kills_detached_worker_after_launcher_exits(tmp_path):
    import psutil

    ready = tmp_path / "detached-worker.pid"
    child_code = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    root = psutil.Process(process.pid)
    owned = {(root.pid, root.create_time()): root}
    try:
        timeout = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < timeout:
            time.sleep(0.01)
        assert ready.exists()
        detached = psutil.Process(int(ready.read_text()))
        remember_descendants(owned)
        assert any(child.pid == detached.pid for child in owned.values())
        root.terminate()
        process.wait(timeout=5)
        assert detached.is_running()
        stop_process_tree(process, owned, grace_seconds=0.1)
        assert not detached.is_running() or detached.status() == psutil.STATUS_ZOMBIE
    finally:
        stop_process_tree(process, owned, grace_seconds=0.1)


def test_hard_watchdog_covers_child_setup_before_torchrun(monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def blocked_setup(command, **options):
        assert "--execute-started" in command and "--worker-plan" not in command
        assert options["start_new_session"] is True
        process = real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **options)
        children.append(process)
        return process

    monkeypatch.setattr("scripts.train_segment.subprocess.Popen", blocked_setup)
    monkeypatch.setattr(
        "scripts.train_segment.segment_budget", lambda args: SimpleNamespace(seconds=30.2)
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        supervise(SimpleNamespace(plan_only=False), ["--fixture-setup"])
    assert time.monotonic() - started < 10
    assert len(children) == 1 and children[0].poll() is not None

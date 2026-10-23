"""Bounded segment configuration, verified S3 handoffs and real Lightning resume."""

import errno
import io
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from hydra.utils import instantiate
from lightning.pytorch.strategies import DDPStrategy
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
    execute_segment,
    freeze_naturalistic_manifest,
    main,
    make_config,
    remember_descendants,
    stop_process_tree,
    streaming_acceptance,
    supervise,
    validate_launch_options,
    verify_resume_state,
    verify_streaming_report,
    wait_owned_processes,
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


def streaming_report(*, temporal=True):
    from test_hierarchical_dynamics import model_factory

    from scripts.benchmark_streaming_gpu import architecture_facts

    architecture = architecture_facts(
        model_factory(
            aggregation_windows=[1, 2, 2] if temporal else [1, 1, 1],
            temporal_variance_weight=0.1 if temporal else 0.0,
        )
    )
    losses = {
        "temporal_variance_loss": 0.05,
        "level_1/temporal_variance_loss": 0.02,
        "level_2/temporal_variance_loss": 0.03,
    }
    common = {
        "world_size": 4,
        "architecture": architecture,
        "optimizer_steps": 400,
        "ema_updates": 400,
        "frozen_index_sha256": "a" * 64,
    }
    return {
        "ok": True,
        "model_variant": "temporal_aggregation" if temporal else "pointwise",
        "architecture": architecture,
        "data_worker_start_method": "spawn",
        "workers_per_rank": 1,
        "cache_refills_observed_on_all_ranks": True,
        "durable_checkpoint_required": True,
        "frozen_index": {
            "source_index_sha256": "a" * 64,
            "frozen_index_sha256": "a" * 64,
            "status": "complete",
            "verified_sources": 889,
            "split_counts": {"val": 130},
            "validation_clips": 130,
            "validation_used_clips": 128,
            "validation_excluded_tail_clips": 2,
        },
        "checkpoint": {
            "strict_tensor_and_metadata_roundtrip": True,
            "bytes": 1024,
            "architecture": architecture,
            "durable_s3": {
                "architecture": architecture,
                "complete_remote_readback_verified": True,
                "download_sha256_verified": True,
                "strict_tensor_and_metadata_roundtrip": True,
                "receipt": {
                    "owner": {"index_sha256": "a" * 64},
                    "checkpoint": {"step": 400, "bytes": 1024},
                },
            },
        },
        "ranks": [
            {
                **common,
                "rank": rank,
                "global_batch": 64,
                "online_weights_changed": True,
                "all_model_parameters_finite": True,
                "data_worker_start_method": "spawn",
                "workers_per_rank": 1,
                "cache_refills_observed_after_warmup": True,
                "rank_cache_at_train_end": {"verified_files": 4},
                "rank_cache_at_warmup": {"verified_files": 1},
                "temporal_training_samples": [{"optimizer_step": 1, "metrics": losses}],
            }
            for rank in range(4)
        ],
        "validation": [
            {
                **common,
                "rank": rank,
                "clips": 32,
                "global_used_clips": 128,
                "global_excluded_tail_clips": 2,
                "metrics": {
                    "val/loss": 1.0,
                    **{f"val/{key}": value for key, value in losses.items()},
                },
            }
            for rank in range(4)
        ],
    }


@pytest.mark.parametrize("temporal", [False, True])
def test_hardware_gate_accepts_only_matching_model_and_complete_corpus(tmp_path, temporal):
    path = tmp_path / "streaming.json"
    path.write_text(json.dumps(streaming_report(temporal=temporal)))
    assert (
        len(verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=temporal)) == 64
    )
    with pytest.raises(ValueError, match="streaming probe"):
        verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=not temporal)
    with pytest.raises(ValueError, match="streaming probe"):
        verify_streaming_report(path, expected_index_sha256="b" * 64, temporal=temporal)


@pytest.mark.parametrize(
    "damage",
    [
        "fork",
        "missing_validation",
        "partial_validation",
        "partial_index",
        "wrong_sha",
        "pointwise_geometry",
        "no_floor",
        "no_temporal_samples",
        "nonfinite_temporal",
        "no_remote_readback",
        "wrong_checkpoint_index",
        "stale_rank",
        "no_rank_refill",
        "nonfinite_validation",
    ],
)
def test_temporal_gate_rejects_incomplete_or_mismatched_hardware_evidence(tmp_path, damage):
    report = streaming_report()
    if damage == "fork":
        report["data_worker_start_method"] = "fork"
    elif damage == "missing_validation":
        report.pop("validation")
    elif damage == "partial_validation":
        report["validation"][0]["clips"] = 16
    elif damage == "partial_index":
        report["frozen_index"]["status"] = "partial"
    elif damage == "wrong_sha":
        report["frozen_index"]["frozen_index_sha256"] = "b" * 64
    elif damage == "pointwise_geometry":
        report["architecture"]["aggregation_windows"] = [1, 1, 1]
    elif damage == "no_floor":
        report["architecture"]["temporal_variance_weight"] = 0
    elif damage == "no_temporal_samples":
        report["ranks"][2]["temporal_training_samples"] = []
    elif damage == "nonfinite_temporal":
        report["ranks"][2]["temporal_training_samples"][0]["metrics"]["temporal_variance_loss"] = (
            float("nan")
        )
    elif damage == "no_remote_readback":
        report["checkpoint"]["durable_s3"]["complete_remote_readback_verified"] = False
    elif damage == "wrong_checkpoint_index":
        report["checkpoint"]["durable_s3"]["receipt"]["owner"]["index_sha256"] = "b" * 64
    elif damage == "stale_rank":
        report["ranks"][1]["ema_updates"] = 399
    elif damage == "no_rank_refill":
        report["ranks"][3]["rank_cache_at_train_end"]["verified_files"] = 1
    elif damage == "nonfinite_validation":
        report["validation"][1]["metrics"]["val/loss"] = float("nan")
    path = tmp_path / "streaming.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="streaming probe"):
        verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=True)


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
        "resume_step": 1000,
        "resolved_data_seed": 1042,
    }


@pytest.mark.parametrize("temporal", [False, True])
def test_production_config_preserves_objective_global_batch_and_absolute_schedule(
    tmp_path, temporal
):
    cfg = make_config(plan(tmp_path) | {"temporal": temporal})
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
    assert list(cfg.callbacks) == ["timer", "model_checkpoint", "progress"]
    assert cfg.callbacks.model_checkpoint.first_checkpoint_step == 1250
    assert cfg.callbacks.model_checkpoint.async_upload is True
    assert cfg.callbacks.model_checkpoint.dirpath == str(tmp_path / "output" / "checkpoints")
    assert cfg.callbacks.progress.every_n_steps == 100
    assert cfg.callbacks.progress.all_ranks is True
    assert cfg.data.download_timeout_seconds == 300
    assert cfg.data.worker_timeout_seconds == 360
    assert cfg.data.worker_timeout_seconds < cfg.trainer.strategy.timeout.seconds
    assert list(cfg.model.aggregation_windows) == ([1, 2, 2] if temporal else [1, 1, 1])
    assert cfg.model.temporal_variance_weight == (0.1 if temporal else 0)
    logger = instantiate(cfg.logger.csv)
    assert Path(logger.log_dir) == tmp_path / "output" / "metrics" / "training"
    logger.log_metrics({"train/level_2/temporal_variance_loss": 0.08}, step=50)
    logger.save()
    assert "temporal_variance_loss" in (Path(logger.log_dir) / "metrics.csv").read_text()
    assert cfg.model.backbone.config.compact_masked_tokens is True


@pytest.mark.parametrize(
    ("resume_step", "first_checkpoint_step"),
    [(0, 250), (250, 500), (4750, 5000), (374900, 375000), (374999, 375000)],
)
def test_clean_segment_checks_publication_early_after_resume(
    tmp_path, resume_step, first_checkpoint_step
):
    cfg = make_config(plan(tmp_path) | {"clean_binaural": True, "resume_step": resume_step})
    assert cfg.callbacks.model_checkpoint.first_checkpoint_step == first_checkpoint_step
    assert resume_step < first_checkpoint_step <= cfg.trainer.max_steps
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 5000
    assert cfg.trainer.max_steps == cfg.model.lr_total_steps == 375000


@pytest.mark.parametrize(
    ("transfer_seconds", "collective_seconds"), [(30, 180), (600, 720), (900.25, 1021)]
)
def test_segment_collective_timeout_covers_configured_publication_budget(
    tmp_path, transfer_seconds, collective_seconds
):
    segment = plan(tmp_path)
    segment["budget"]["transfer_timeout_seconds"] = transfer_seconds
    cfg = make_config(segment)
    strategy = instantiate(cfg.trainer.strategy)
    assert isinstance(strategy, DDPStrategy)
    assert strategy._timeout == timedelta(seconds=collective_seconds)
    assert cfg.callbacks.model_checkpoint.transfer_timeout_seconds == transfer_seconds
    assert cfg.trainer.strategy.find_unused_parameters is False


def test_segment_preserves_a_longer_configured_collective_timeout(tmp_path, monkeypatch):
    import hydra

    compose = hydra.compose

    def longer_timeout(**kwargs):
        cfg = compose(**kwargs)
        cfg.trainer.strategy.timeout.seconds = 1200
        return cfg

    monkeypatch.setattr(hydra, "compose", longer_timeout)
    strategy = instantiate(make_config(plan(tmp_path)).trainer.strategy)
    assert strategy._timeout == timedelta(seconds=1200)


def test_regular_h100_preset_retains_short_collective_timeout(request):
    cfg = request.getfixturevalue("make_config")(["experiment=audioset_h100_four"])
    strategy = instantiate(cfg.trainer.strategy)
    assert strategy._timeout == timedelta(seconds=180)
    assert SegmentBudget(10, 160, 15.12).report()["minimum_ddp_timeout_seconds"] == 720


@pytest.mark.parametrize("transfer_seconds", [0, -1, float("inf"), float("nan")])
def test_segment_rejects_invalid_transfer_timeout(tmp_path, transfer_seconds):
    segment = plan(tmp_path)
    segment["budget"]["transfer_timeout_seconds"] = transfer_seconds
    with pytest.raises(ValueError, match="transfer timeout must be finite and positive"):
        make_config(segment)


@pytest.mark.parametrize("collective_seconds", [180, 600, 719, None])
def test_durable_callback_rejects_short_or_implicit_ddp_timeout_before_setup(
    tmp_path, collective_seconds, monkeypatch
):
    from src.callbacks.step_checkpoint import StepCheckpoint

    callback = instantiate(make_config(plan(tmp_path)).callbacks.model_checkpoint)
    strategy = DDPStrategy(
        timeout=None if collective_seconds is None else timedelta(seconds=collective_seconds)
    )

    def unexpected_setup(*args, **kwargs):
        pytest.fail("Unsafe callback must fail before checkpoint setup collectives")

    monkeypatch.setattr(StepCheckpoint, "setup", unexpected_setup)
    with pytest.raises(ValueError, match="collective timeout must be at least 720s"):
        callback.setup(SimpleNamespace(strategy=strategy), None, "fit")


def test_durable_callback_accepts_segment_timeout_and_runs_checkpoint_setup(tmp_path, monkeypatch):
    from src.callbacks.step_checkpoint import StepCheckpoint

    cfg = make_config(plan(tmp_path))
    callback = instantiate(cfg.callbacks.model_checkpoint)
    trainer = SimpleNamespace(strategy=instantiate(cfg.trainer.strategy))
    calls = []
    monkeypatch.setattr(StepCheckpoint, "setup", lambda *args: calls.append(args))
    callback.setup(trainer, None, "fit")
    assert calls == [(callback, trainer, None, "fit")]


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


@pytest.mark.parametrize("unsupported_pidfd", [False, True])
def test_watchdog_kills_detached_worker_after_launcher_exits(
    tmp_path, monkeypatch, unsupported_pidfd
):
    import psutil

    wait_calls = []
    if unsupported_pidfd:

        def unsupported_wait(*args, **kwargs):
            wait_calls.append(1)
            raise OSError(errno.EINVAL, "pidfd_open rejected by provider kernel")

        monkeypatch.setattr(psutil, "wait_procs", unsupported_wait)
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
        if unsupported_pidfd:
            assert len(wait_calls) == 2
    finally:
        stop_process_tree(process, owned, grace_seconds=0.1)


def test_process_wait_fallback_is_bounded_and_respects_reused_pids(monkeypatch):
    import psutil

    def unsupported_wait(*args, **kwargs):
        raise OSError(errno.EINVAL, "unsupported pidfd")

    monkeypatch.setattr(psutil, "wait_procs", unsupported_wait)
    reused = SimpleNamespace(
        is_running=lambda: False, status=lambda: pytest.fail("reused PID inspected")
    )
    live = SimpleNamespace(is_running=lambda: True, status=lambda: psutil.STATUS_SLEEPING)
    zombie = SimpleNamespace(is_running=lambda: True, status=lambda: psutil.STATUS_ZOMBIE)
    started = time.monotonic()
    gone, alive = wait_owned_processes([reused, live, zombie], timeout=0.1)
    assert alive == [live]
    assert gone == [reused, zombie]
    assert 0.08 <= time.monotonic() - started < 1


def test_process_wait_does_not_swallow_other_os_errors(monkeypatch):
    import psutil

    def forbidden_wait(*args, **kwargs):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(psutil, "wait_procs", forbidden_wait)
    with pytest.raises(OSError) as error:
        wait_owned_processes([], timeout=0.1)
    assert error.value.errno == errno.EACCES


def test_process_cleanup_rejects_a_surviving_writer_under_pidfd_fallback(monkeypatch):
    import psutil

    def unsupported_wait(*args, **kwargs):
        raise OSError(errno.EINVAL, "unsupported pidfd")

    monkeypatch.setattr(psutil, "wait_procs", unsupported_wait)
    alive = SimpleNamespace(
        suspend=lambda: None,
        terminate=lambda: None,
        resume=lambda: None,
        kill=lambda: None,
        children=lambda **kwargs: [],
        is_running=lambda: True,
        status=lambda: psutil.STATUS_SLEEPING,
    )
    launcher = SimpleNamespace(wait=lambda **kwargs: pytest.fail("unverified writer accepted"))
    with pytest.raises(RuntimeError, match="survived"):
        stop_process_tree(launcher, {(10, 100.0): alive}, grace_seconds=0.01)


@pytest.mark.parametrize("unsupported_pidfd", [False, True])
def test_hard_watchdog_covers_child_setup_before_torchrun(monkeypatch, unsupported_pidfd):
    if unsupported_pidfd:

        def unsupported_wait(*args, **kwargs):
            raise OSError(errno.EINVAL, "unsupported pidfd")

        monkeypatch.setattr("psutil.wait_procs", unsupported_wait)
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


def test_process_wait_fallback_keeps_original_deadline(monkeypatch):
    import psutil

    clock = [100.0]

    def late_unsupported_wait(*args, **kwargs):
        clock[0] += kwargs["timeout"]
        raise OSError(errno.EINVAL, "unsupported pidfd")

    monkeypatch.setattr(psutil, "wait_procs", late_unsupported_wait)
    monkeypatch.setattr("scripts.train_segment.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "scripts.train_segment.time.sleep", lambda seconds: pytest.fail("deadline reset")
    )
    live = SimpleNamespace(is_running=lambda: True, status=lambda: psutil.STATUS_SLEEPING)
    assert wait_owned_processes([live], timeout=0.1) == ([], [live])


def test_segment_runner_requires_explicit_nat_production_selection(tmp_path):
    report = streaming_report(temporal=True)
    report["model_variant"] = "nat_temporal_aggregation"
    report["architecture"].update(
        input_channels=2, masked_crops_per_recording=8, masked_crop_samples=32160
    )
    path = tmp_path / "nat-streaming.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="explicit --nat"):
        verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=True)


@pytest.mark.parametrize("temporal", [False, True])
def test_production_gate_preserves_clean_pre_nat_mono_receipts(tmp_path, temporal):
    report = streaming_report(temporal=temporal)
    architecture_records = [
        report,
        report["checkpoint"],
        report["checkpoint"]["durable_s3"],
        *report["ranks"],
        *report["validation"],
    ]
    for row in architecture_records:
        for key in ("input_channels", "masked_crops_per_recording", "masked_crop_samples"):
            row["architecture"].pop(key, None)
    path = tmp_path / "legacy-mono-report.json"
    path.write_text(json.dumps(report))
    assert (
        len(verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=temporal)) == 64
    )
    report["ranks"][2]["architecture"] = report["ranks"][2]["architecture"] | {"input_channels": 2}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="streaming probe"):
        verify_streaming_report(path, expected_index_sha256="a" * 64, temporal=temporal)


def nat_launch_args(tmp_path, **overrides):
    return SimpleNamespace(
        **{
            "nat": True,
            "temporal": False,
            "plan_only": False,
            "naturalistic_manifest": tmp_path / "assets.json",
            "naturalistic_manifest_sha256": "b" * 64,
            "streaming_smoke_report": None,
            "index_sha256": "a" * 64,
            "skip_streaming_acceptance": True,
            "streaming_acceptance_waiver_reason": "User explicitly requested Nat training without a separate GPU preflight.",
            **overrides,
        }
    )


def test_nat_waiver_records_user_reason_and_never_claims_hardware_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.train_segment.verify_streaming_report",
        lambda *args, **kwargs: pytest.fail("A waived report must not be fabricated or verified"),
    )
    result = streaming_acceptance(nat_launch_args(tmp_path))
    assert result["status"] == "waived_by_user"
    assert result["hardware_acceptance_passed"] is False
    assert result["report_sha256"] is None
    assert "User explicitly requested" in result["reason"]
    assert "checkpoint integrity remain enabled" in result["scope"]


@pytest.mark.parametrize(
    "changes",
    [
        {"streaming_acceptance_waiver_reason": None},
        {"streaming_acceptance_waiver_reason": "   "},
        {"skip_streaming_acceptance": False},
        {"streaming_smoke_report": Path("old-mono-report.json")},
        {"naturalistic_manifest": None},
        {"naturalistic_manifest_sha256": "not-a-sha"},
        {"nat": False},
    ],
)
def test_nat_waiver_cannot_silently_bypass_launch_identity(tmp_path, changes):
    with pytest.raises(ValueError):
        validate_launch_options(nat_launch_args(tmp_path, **changes))


def test_hardware_report_remains_required_without_explicit_waiver(tmp_path):
    with pytest.raises(ValueError, match="successful streaming hardware report"):
        streaming_acceptance(
            nat_launch_args(
                tmp_path, skip_streaming_acceptance=False, streaming_acceptance_waiver_reason=None
            )
        )


def test_nat_production_config_preserves_crops_batch_schedule_and_safety_guards(tmp_path):
    from src.utils.runtime import file_sha256

    segment = plan(tmp_path) | {"nat": True, "global_batch": 16}
    work = Path(segment["work_dir"])
    work.mkdir()
    manifest = work / "frozen-naturalistic-manifest.json"
    manifest.write_text('{"pinned": "manifest fixture"}')
    segment["naturalistic_manifest_sha256"] = file_sha256(manifest)
    cfg = make_config(segment)
    validate_config(cfg)
    assert cfg.trainer.devices == cfg.data.batch_size == 4
    assert cfg.model.backbone.config.input_channels == 2
    assert cfg.model.backbone.config.use_gradient_checkpointing is True
    assert cfg.model.backbone.config.compact_masked_tokens is True
    assert cfg.data.masked_crops_per_recording == cfg.model.masked_crops_per_recording == 8
    assert cfg.data.masked_crop_samples == cfg.model.masked_crop_samples == 32160
    assert cfg.data.batch_size * cfg.trainer.devices * cfg.data.masked_crops_per_recording == 128
    assert cfg.data.naturalistic_manifest == str(manifest)
    assert cfg.model.lr_total_steps == cfg.trainer.max_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.model.masked_weight == 1 and cfg.model.temporal_variance_weight == 0.1
    assert cfg.data.allow_partial_index is False and cfg.trainer.limit_val_batches == 1.0
    assert instantiate(cfg.trainer.strategy)._timeout == timedelta(seconds=720)
    assert cfg.callbacks.model_checkpoint.transfer_timeout_seconds == 600
    manifest.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="manifest changed"):
        make_config(segment)


def test_nat_manifest_is_frozen_with_exact_sha_remote_scope_and_training_split(tmp_path):
    from test_naturalistic_audio import _assets, _remote_assets

    from src.utils.runtime import file_sha256

    source, manifest = _assets(tmp_path)
    _remote_assets(source, manifest)
    digest = file_sha256(source)
    destination = tmp_path / "frozen.json"
    assert (
        freeze_naturalistic_manifest(
            source, destination, digest, bucket=manifest["bucket"], endpoint=manifest["endpoint"]
        )
        == digest
    )
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze_naturalistic_manifest(
            source, destination, "c" * 64, bucket=manifest["bucket"], endpoint=manifest["endpoint"]
        )
    with pytest.raises(ValueError, match="same private bucket"):
        freeze_naturalistic_manifest(
            source, destination, digest, bucket="unrelated-bucket", endpoint=manifest["endpoint"]
        )
    manifest["split"] = "test"
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training assets"):
        freeze_naturalistic_manifest(
            source,
            destination,
            file_sha256(source),
            bucket=manifest["bucket"],
            endpoint=manifest["endpoint"],
        )


def test_nat_checkpoint_owner_binds_asset_manifest_and_crop_population(tmp_path):
    client = CheckpointS3()
    repo = repository(client, global_batch=16, naturalistic_manifest_sha256="b" * 64)
    repo.open(initialize=True)
    saved = repo.publish(payload(tmp_path, 2), 2, segment_id="nat-segment-001", data_seed=42)
    assert saved["owner"]["naturalistic_manifest_sha256"] == "b" * 64
    assert saved["owner"]["global_batch"] == 16
    assert saved["owner"]["input_channels"] == 2
    assert saved["owner"]["masked_crops_per_recording"] == 8
    assert saved["owner"]["masked_crop_samples"] == 32160
    for changes in (
        {"global_batch": 16, "naturalistic_manifest_sha256": "c" * 64},
        {"global_batch": 16},
        {},
    ):
        with pytest.raises(ValueError, match="different run"):
            repository(client, **changes).open(initialize=False)
    with pytest.raises(ValueError, match="global batch 16"):
        repository(client, naturalistic_manifest_sha256="b" * 64)


def test_waiver_preserves_completed_corpus_gate_before_checkpoint_or_training(
    tmp_path, monkeypatch
):
    source = tmp_path / "partial-index.json"
    source.write_text("{}")
    args = nat_launch_args(
        tmp_path,
        segment_hours=2,
        budget_usd=40,
        maximum_rate_usd_per_hour=15.12,
        finalization_reserve_seconds=1800,
        transfer_timeout_seconds=600,
        exclusive_checkpoint_writer=True,
        acknowledge_stream_restart=True,
        execute_started=time.time(),
        output=tmp_path / "output",
        work_dir=tmp_path / "work",
        index=source,
        data_endpoint="https://storage.givemeanode.com",
    )
    monkeypatch.setattr(
        "scripts.train_segment.checkpoint_client",
        lambda *args: pytest.fail(
            "An invalid corpus must fail before accessing the checkpoint bucket"
        ),
    )
    monkeypatch.setattr(
        "scripts.train_segment.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("An invalid corpus must never start training"),
    )
    with pytest.raises(ValueError, match="Frozen staging index size/SHA256 mismatch"):
        execute_segment(args)


def test_nat_resume_rejects_changed_manifest_crop_recipe_and_mono_selection(tmp_path):
    from test_hierarchical_dynamics import model_factory
    from test_nat_backbone import nat_config

    from src.models.components.wavjepa_backbone import WavJEPABackbone

    model = model_factory(
        backbone=WavJEPABackbone(nat_config()),
        masked_crops_per_recording=8,
        masked_crop_samples=32160,
        aggregation_windows=[1, 2, 2],
        temporal_variance_weight=0.1,
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
    )
    state = {
        "global_step": 2,
        "state_dict": {"ema_updates": torch.tensor(2)},
        "optimizer_states": [{"state": {0: {"step": torch.tensor(2)}}}],
        "lr_schedulers": [{"last_epoch": 2}],
        "loops": {"fit_loop": {}},
        "ShardedAudioDataModule": {
            "index_sha256": "a" * 64,
            "naturalistic_sha256": "b" * 64,
            "masked_crops_per_recording": 8,
            "masked_crop_samples": 32160,
        },
        model.CHECKPOINT_KEY: model.checkpoint_metadata(),
        model.SCHEDULE_CHECKPOINT_KEY: {
            "total_steps": 375000,
            "warmup_steps": 100000,
            "ema_end_step": 100000,
        },
    }
    receipt = {
        "checkpoint": {"step": 2},
        "owner": {"global_batch": 16, "naturalistic_manifest_sha256": "b" * 64},
    }
    path = tmp_path / "nat-resume.ckpt"
    torch.save(state, path)
    assert (
        verify_resume_state(
            path, receipt, "a" * 64, nat=True, naturalistic_manifest_sha256="b" * 64
        )
        == 2
    )
    for changes in ({}, {"nat": True, "naturalistic_manifest_sha256": "c" * 64}):
        with pytest.raises(ValueError, match="channel/crop/manifest"):
            verify_resume_state(path, receipt, "a" * 64, **changes)
    state["ShardedAudioDataModule"]["masked_crops_per_recording"] = 1
    torch.save(state, path)
    with pytest.raises(ValueError, match="channel/crop/manifest"):
        verify_resume_state(
            path, receipt, "a" * 64, nat=True, naturalistic_manifest_sha256="b" * 64
        )


def test_strict_nat_acceptance_still_requires_matching_remote_assets_and_both_ears(tmp_path):
    report = streaming_report(temporal=True)
    report["model_variant"] = "nat_temporal_aggregation"
    report["architecture"].update(
        input_channels=2, masked_crops_per_recording=8, masked_crop_samples=32160
    )
    report["frozen_index"]["naturalistic_manifest_sha256"] = "b" * 64
    report["checkpoint"]["durable_s3"]["receipt"]["owner"]["global_batch"] = 16
    for rank in report["ranks"]:
        rank.update(
            global_batch=16,
            global_masked_crops=128,
            naturalistic_manifest_sha256="b" * 64,
            nat_asset_streaming={
                "manifest_sha256": "b" * 64,
                "train_end": {
                    "remote_cache_contracts": 1,
                    "brir": {"verified_files": 2, "verified_bytes": 100},
                    "noise": {"verified_files": 2, "verified_bytes": 100},
                },
                "new_verified_brir_objects_after_warmup": 1,
                "new_verified_noise_objects_after_warmup": 1,
            },
            ear_cnn_updates=[
                {
                    "ear": ear,
                    "independent_parameters": True,
                    "parameter_tensors": 1,
                    "changed_parameter_tensors": 1,
                    "all_parameters_finite": True,
                }
                for ear in (0, 1)
            ],
        )
    for row in report["validation"]:
        row["naturalistic_manifest_sha256"] = "b" * 64
    path = tmp_path / "nat-passed.json"
    path.write_text(json.dumps(report))
    args = nat_launch_args(
        tmp_path,
        skip_streaming_acceptance=False,
        streaming_acceptance_waiver_reason=None,
        streaming_smoke_report=path,
    )
    result = streaming_acceptance(args)
    assert result["hardware_acceptance_passed"] is True and result["status"] == "passed"
    report["ranks"][2]["ear_cnn_updates"][1]["changed_parameter_tensors"] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="both independent ear CNNs"):
        streaming_acceptance(args)


def clean_curriculum_state():
    from test_hierarchical_dynamics import model_factory
    from test_nat_backbone import nat_config

    from src.models.components.wavjepa_backbone import WavJEPABackbone

    model = model_factory(
        backbone=WavJEPABackbone(nat_config()),
        masked_crops_per_recording=8,
        masked_crop_samples=32160,
        aggregation_windows=[1, 2, 2],
        temporal_variance_weight=0.1,
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
    )
    return {
        "global_step": 2,
        "state_dict": {"ema_updates": torch.tensor(2)},
        "optimizer_states": [
            {"state": {0: {"step": torch.tensor(2), "exp_avg": torch.tensor([0.7])}}}
        ],
        "lr_schedulers": [{"last_epoch": 2}],
        "loops": {"fit_loop": {"optimizer_step": 2}},
        "ShardedAudioDataModule": {
            "index_sha256": "a" * 64,
            "naturalistic_sha256": None,
            "data_mode": "clean_binaural",
            "masked_crops_per_recording": 8,
            "masked_crop_samples": 32160,
        },
        model.CHECKPOINT_KEY: model.checkpoint_metadata(),
        model.SCHEDULE_CHECKPOINT_KEY: {
            "total_steps": 375000,
            "warmup_steps": 100000,
            "ema_end_step": 100000,
        },
    }


def test_clean_binaural_config_uses_unchanged_nat_model_without_manifest(tmp_path):
    clean = make_config(plan(tmp_path) | {"clean_binaural": True})
    assert clean.data.clean_binaural is True
    assert clean.data.naturalistic_manifest is None
    assert clean.data.batch_size == 4 and clean.trainer.devices == 4
    assert clean.data.masked_crops_per_recording == 8
    assert clean.data.masked_crop_samples == 32160
    assert clean.model.backbone.config.input_channels == 2
    assert clean.model.lr_total_steps == 375000
    segment = plan(tmp_path) | {
        "clean_binaural": True,
        "nat_readiness": {
            "endpoint": "https://storage.givemeanode.com",
            "bucket": "audio",
            "key": "nat/ready.json",
        },
    }
    cfg = make_config(segment)
    assert cfg.callbacks.nat_readiness.ready_key == "nat/ready.json"


@pytest.mark.parametrize(
    "change",
    [
        {"nat": True},
        {"naturalistic_manifest": Path("manifest.json")},
        {"curriculum_source_prefix": "training/other-run"},
        {"stop_on_nat_ready_key": "nat/ready.json"},
    ],
)
def test_clean_curriculum_launch_rejects_ambiguous_or_incomplete_phases(tmp_path, change):
    args = nat_launch_args(
        tmp_path,
        nat=False,
        clean_binaural=True,
        naturalistic_manifest=None,
        naturalistic_manifest_sha256=None,
    )
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        validate_launch_options(args)


def test_clean_handoff_seals_writer_and_imports_unchanged_full_state(tmp_path):
    from scripts.train_segment import import_clean_curriculum
    from src.data.sharded_audio_datamodule import data_state_sha256
    from src.utils.runtime import file_sha256

    client = CheckpointS3()
    repo = repository(client, clean_binaural=True, global_batch=16)
    repo.open(initialize=True)
    original = tmp_path / "original.ckpt"
    state = clean_curriculum_state()
    torch.save(state, original)
    receipt = repo.publish(original, 2, segment_id="segment-clean", data_seed=42)
    with pytest.raises(ValueError, match="exited writer"):
        repo.publish_clean_handoff(receipt, writer_finished=False)
    handoff = repo.publish_clean_handoff(receipt, writer_finished=True)
    assert handoff["ready_for_nat"] is True
    for action in (
        lambda: repo.open(initialize=False),
        lambda: repo.publish(payload(tmp_path, 4), 4, segment_id="segment-late", data_seed=44),
    ):
        with pytest.raises(ValueError, match="sealed"):
            action()
    destination = tmp_path / "imported.ckpt"
    step, imported_receipt, transition = import_clean_curriculum(
        repo,
        destination,
        expected_checkpoint_sha256=file_sha256(original),
        index_sha256="a" * 64,
        target_manifest_sha256="b" * 64,
    )
    assert step == 2 and imported_receipt == receipt
    assert destination.read_bytes() == original.read_bytes()
    assert transition == {
        "source_checkpoint_sha256": file_sha256(original),
        "source_data_state_sha256": data_state_sha256(state["ShardedAudioDataModule"]),
        "index_sha256": "a" * 64,
        "target_manifest_sha256": "b" * 64,
    }
    restored = torch.load(destination, weights_only=True)
    assert restored["optimizer_states"][0]["state"][0]["exp_avg"].item() == pytest.approx(0.7)
    assert restored["lr_schedulers"][0]["last_epoch"] == 2
    assert restored["loops"] == state["loops"]
    with pytest.raises(ValueError, match="pinned source"):
        repo.clean_handoff("c" * 64)
    client.corrupt_checkpoint = True
    with pytest.raises(ValueError, match="SHA256"):
        import_clean_curriculum(
            repo,
            tmp_path / "corrupt.ckpt",
            expected_checkpoint_sha256=file_sha256(original),
            index_sha256="a" * 64,
            target_manifest_sha256="b" * 64,
        )


@pytest.mark.parametrize("damage", ["mono", "nat", "crop", "optimizer", "ema", "schedule"])
def test_clean_import_rejects_incompatible_state_even_with_valid_blob_checksum(tmp_path, damage):
    from scripts.train_segment import import_clean_curriculum

    state = clean_curriculum_state()
    data = state["ShardedAudioDataModule"]
    if damage in {"mono", "nat"}:
        data["data_mode"] = damage
    elif damage == "crop":
        data["masked_crop_samples"] = 32000
    elif damage == "optimizer":
        state["optimizer_states"][0]["state"][0]["step"] = torch.tensor(1)
    elif damage == "ema":
        state["state_dict"]["ema_updates"] = torch.tensor(1)
    elif damage == "schedule":
        state["lr_schedulers"][0]["last_epoch"] = 1
    original = tmp_path / "invalid.ckpt"
    torch.save(state, original)
    repo = repository(CheckpointS3(), clean_binaural=True, global_batch=16)
    repo.open(initialize=True)
    receipt = repo.publish(original, 2, segment_id="segment-clean", data_seed=42)
    repo.publish_clean_handoff(receipt, writer_finished=True)
    with pytest.raises(ValueError, match="recipe|complete matching"):
        import_clean_curriculum(
            repo,
            tmp_path / "imported.ckpt",
            expected_checkpoint_sha256=receipt["checkpoint"]["sha256"],
            index_sha256="a" * 64,
            target_manifest_sha256="b" * 64,
        )


def test_nat_destination_binds_clean_source_lineage(tmp_path):
    client = CheckpointS3()
    source = {
        "run_id": "source-run-001",
        "prefix": "training/source-run-001",
        "checkpoint_sha256": "c" * 64,
    }
    repo = repository(
        client, global_batch=16, naturalistic_manifest_sha256="b" * 64, curriculum_source=source
    )
    repo.open(initialize=True)
    repo.publish(payload(tmp_path, 4), 4, segment_id="segment-nat", data_seed=44)
    assert repo.owner["curriculum_source"] == source
    with pytest.raises(ValueError, match="different run"):
        repository(client, global_batch=16, naturalistic_manifest_sha256="b" * 64).open(
            initialize=False
        )

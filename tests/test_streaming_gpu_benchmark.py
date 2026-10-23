"""CPU checks for the GPU probe's real config, lifecycle and success evidence."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from hydra.utils import instantiate
from test_stage_audioset import FakeS3
from torch.utils.data import DataLoader

from scripts.benchmark_streaming_gpu import (
    StreamingTimings,
    architecture_facts,
    collective_timeout_seconds,
    durable_checkpoint_options,
    durable_checkpoint_roundtrip,
    finite_metrics,
    make_config,
    open_checkpoint_repository,
    summarize,
    temporal_loss_metrics,
    verify_checkpoint,
    worker_facts,
)
from src.utils.runtime import validate_config


def durable_args(tmp_path, **changes):
    return SimpleNamespace(
        **{
            "output": tmp_path,
            "endpoint": "https://storage.example.invalid",
            "devices": 4,
            "workers": 1,
            "steps": 400,
            "warmup": 25,
            "compact": True,
            "checkpoint_bucket": "checkpoint-probes",
            "checkpoint_prefix": "probes/streaming-probe-001",
            "checkpoint_run_id": "streaming-probe-001",
            "checkpoint_endpoint": "https://storage.example.invalid",
            "checkpoint_transfer_timeout_seconds": 600,
            "exclusive_checkpoint_writer": True,
            **changes,
        }
    )


def test_worker_evidence_reads_actual_loader_context():
    loader = DataLoader([1, 2], num_workers=1, multiprocessing_context="spawn")
    assert worker_facts(loader) == {
        "data_worker_start_method": "spawn",
        "workers_per_rank": 1,
    }
    with pytest.raises(RuntimeError, match="spawned input workers"):
        worker_facts(DataLoader([1, 2], num_workers=1))


@pytest.mark.parametrize("devices", [2, 4])
@pytest.mark.parametrize("temporal", [False, True])
def test_streaming_gpu_config_preserves_global_batch_objective_and_full_schedule(
    tmp_path, devices, temporal
):
    args = SimpleNamespace(
        output=tmp_path,
        endpoint="https://storage.givemeanode.com",
        devices=devices,
        workers=1,
        steps=400,
        warmup=25,
        compact=True,
        temporal=temporal,
    )
    cfg = make_config(
        args, {"path": "/tmp/frozen.json", "frozen_index_sha256": "a" * 64}, tmp_path / "cache"
    )
    validate_config(cfg)
    assert cfg.data._target_ == "src.data.sharded_audio_datamodule.ShardedAudioDataModule"
    assert cfg.data.batch_size * cfg.trainer.devices == 64
    assert cfg.data.num_workers == 1 and cfg.data.allow_partial_index is True
    assert cfg.data.cache_max_gib == 40 and cfg.trainer.use_distributed_sampler is False
    assert cfg.model.backbone.config.compact_masked_tokens is True
    assert cfg.model.backbone.config.use_gradient_checkpointing is False
    assert cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.model.regularization_weight == 0.1 and cfg.trainer.gradient_clip_val == 1.0
    assert cfg.model.masked_weight == 1.0
    assert list(cfg.model.aggregation_windows) == ([1, 2, 2] if temporal else [1, 1, 1])
    assert cfg.model.temporal_variance_weight == (0.1 if temporal else 0.0)
    assert cfg.callbacks.timing.temporal is temporal
    assert cfg.trainer.limit_val_batches == 0 and cfg.test is False


@pytest.mark.parametrize("temporal", [False, True])
def test_full_checkpoint_after_validation_preserves_training_and_data_state(
    tmp_path, monkeypatch, temporal
):
    from test_hierarchical_dynamics import model_factory

    class FixtureData(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader([{"waveform": torch.randn(512 * 9)} for _ in range(4)], batch_size=2)

        val_dataloader = train_dataloader

        def state_dict(self):
            return {"index_sha256": "a" * 64, "stream_position_checkpointed": False}

    steps = 51 if temporal else 2
    model = model_factory(
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
        aggregation_windows=[1, 2, 2] if temporal else [1, 1, 1],
        temporal_variance_weight=0.1 if temporal else 0.0,
        training_metric_interval=50 if temporal else 1,
    )
    data = FixtureData()
    # Exercise the real callback ordering/metric names on CPU. Only CUDA timing
    # and the separately tested worker inventory are substituted here.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(
        "scripts.benchmark_streaming_gpu.worker_facts",
        lambda loader: {"workers_per_rank": 1, "data_worker_start_method": "spawn"},
    )
    timing = StreamingTimings(
        tmp_path,
        tmp_path / "cache",
        steps=steps,
        warmup=1,
        frozen_sha256="a" * 64,
        temporal=temporal,
    )
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=steps,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        callbacks=[timing],
    )
    trainer.fit(model, datamodule=data)
    if temporal:
        assert temporal_loss_metrics(trainer.callback_metrics, stage="train")
        assert [sample["optimizer_step"] for sample in timing.temporal_samples] == [1, 50]
    rank_report = json.loads((tmp_path / "rank-0.json").read_text())
    assert rank_report["architecture"] == architecture_facts(model)
    assert rank_report["temporal_training_samples"] == timing.temporal_samples
    trainer.limit_val_batches = 1.0
    metrics = finite_metrics(trainer.validate(model=model, datamodule=data, verbose=False))
    assert metrics["val/loss"] > 0
    if temporal:
        assert temporal_loss_metrics(metrics, stage="val")
    path = tmp_path / "probe.ckpt"
    trainer.save_checkpoint(path, weights_only=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    geometry = verify_checkpoint(checkpoint, model, data, steps)
    assert geometry["level_support_windows"] == ([1, 2, 4] if temporal else [1, 1, 1])
    assert geometry["temporal_variance_weight"] == (0.1 if temporal else 0.0)
    client = FakeS3()
    monkeypatch.setattr(
        "scripts.benchmark_streaming_gpu.checkpoint_client", lambda endpoint: client
    )
    repository = open_checkpoint_repository(
        durable_args(tmp_path), {"frozen_index_sha256": "a" * 64}
    )
    report = durable_checkpoint_roundtrip(repository, path, model, data, steps, 600)
    assert report["complete_remote_readback_verified"] is True
    assert report["download_sha256_verified"] is True
    assert report["strict_tensor_and_metadata_roundtrip"] is True
    assert report["architecture"] == geometry
    assert report["receipt"]["checkpoint"]["step"] == steps
    assert report["receipt"]["checkpoint"]["bytes"] == path.stat().st_size
    assert client.objects[report["receipt"]["checkpoint"]["key"]] == path.read_bytes()
    assert not path.with_name("downloaded-checkpoint.ckpt").exists()
    for name in (
        "upload_and_full_readback_seconds",
        "download_and_sha256_seconds",
        "downloaded_tensor_and_metadata_verify_seconds",
        "total_seconds",
    ):
        assert report[name] > 0
    if temporal:
        checkpoint[model.CHECKPOINT_KEY]["dynamics"]["aggregation_windows"] = [1, 1, 1]
        with pytest.raises(ValueError, match="configuration mismatch"):
            verify_checkpoint(checkpoint, model, data, steps)
        checkpoint[model.CHECKPOINT_KEY]["dynamics"]["aggregation_windows"] = [1, 2, 2]
        checkpoint[model.CHECKPOINT_KEY]["dynamics"]["temporal_variance_weight"] = 0.0
        with pytest.raises(ValueError, match="configuration mismatch"):
            verify_checkpoint(checkpoint, model, data, steps)
        checkpoint[model.CHECKPOINT_KEY]["dynamics"]["temporal_variance_weight"] = 0.1
    checkpoint[type(data).__qualname__] = {"index_sha256": "wrong"}
    with pytest.raises(RuntimeError, match="data-index"):
        verify_checkpoint(checkpoint, model, data, steps)


@pytest.mark.parametrize("devices", [2, 4])
@pytest.mark.parametrize("temporal", [False, True])
def test_summary_requires_matching_all_rank_training_validation_and_index(
    tmp_path, devices, temporal
):
    from test_hierarchical_dynamics import model_factory

    geometry = architecture_facts(
        model_factory(
            aggregation_windows=[1, 2, 2] if temporal else [1, 1, 1],
            temporal_variance_weight=0.1 if temporal else 0.0,
        )
    )
    samples = (
        [
            {
                "optimizer_step": 1,
                "metrics": {
                    "temporal_variance_loss": 0.05,
                    "level_1/temporal_variance_loss": 0.02,
                    "level_2/temporal_variance_loss": 0.03,
                },
            }
        ]
        if temporal
        else []
    )
    frozen = {"frozen_index_sha256": "a" * 64, "validation_used_clips": 128}
    validation_metrics = {"val/loss": 1.0}
    if temporal:
        validation_metrics.update(
            {f"val/{key}": value for key, value in samples[0]["metrics"].items()}
        )
    for rank in range(devices):
        identity = {
            "rank": rank,
            "world_size": devices,
            "optimizer_steps": 400,
            "ema_updates": 400,
            "architecture": geometry,
            **frozen,
        }
        (tmp_path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    **identity,
                    "online_weights_changed": True,
                    "data_worker_start_method": "spawn",
                    "workers_per_rank": 1,
                    "all_model_parameters_finite": True,
                    "cache_refills_observed_after_warmup": True,
                    "rank_cache_at_warmup": {"verified_files": 1},
                    "rank_cache_at_train_end": {"verified_files": 3},
                    "mean_seconds_per_update": 0.5 + rank * 0.1,
                    "temporal_training_samples": samples,
                }
            )
        )
        (tmp_path / f"validation-rank-{rank}.json").write_text(
            json.dumps({**identity, "clips": 128 // devices, "metrics": validation_metrics})
        )
    checkpoint = {
        "strict_tensor_and_metadata_roundtrip": True,
        "bytes": 1024,
        "architecture": geometry,
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    if devices == 4:
        with pytest.raises(RuntimeError, match="Durable S3 checkpoint"):
            summarize(
                tmp_path, devices, 400, frozen, require_durable_checkpoint=True, temporal=temporal
            )
        durable = {
            "complete_remote_readback_verified": True,
            "download_sha256_verified": True,
            "strict_tensor_and_metadata_roundtrip": True,
            "architecture": geometry,
            "receipt": {
                "owner": {"index_sha256": frozen["frozen_index_sha256"]},
                "checkpoint": {"step": 400, "bytes": 1024},
            },
        }
        checkpoint["durable_s3"] = durable
        checkpoint_path.write_text(json.dumps(checkpoint))
        assert summarize(
            tmp_path, devices, 400, frozen, require_durable_checkpoint=True, temporal=temporal
        )["ok"]
        for field in (
            "complete_remote_readback_verified",
            "download_sha256_verified",
            "strict_tensor_and_metadata_roundtrip",
        ):
            durable[field] = False
            checkpoint_path.write_text(json.dumps(checkpoint))
            with pytest.raises(RuntimeError, match="Durable S3 checkpoint"):
                summarize(
                    tmp_path,
                    devices,
                    400,
                    frozen,
                    require_durable_checkpoint=True,
                    temporal=temporal,
                )
            durable[field] = True
        checkpoint_path.write_text(json.dumps(checkpoint))
    result = summarize(
        tmp_path, devices, 400, frozen, require_durable_checkpoint=devices == 4, temporal=temporal
    )
    assert result["model_variant"] == ("temporal_aggregation" if temporal else "pointwise")
    assert result["seconds_per_update"] == pytest.approx(0.5 + (devices - 1) * 0.1)
    train_path = tmp_path / "rank-1.json"
    original_train = train_path.read_text()
    no_refill = json.loads(original_train)
    no_refill["rank_cache_at_train_end"]["verified_files"] = 1
    train_path.write_text(json.dumps(no_refill))
    with pytest.raises(RuntimeError, match="additional verified shard downloads"):
        summarize(tmp_path, devices, 400, frozen, temporal=temporal)
    train_path.write_text(original_train)
    wrong_workers = json.loads(original_train)
    wrong_workers["data_worker_start_method"] = "fork"
    train_path.write_text(json.dumps(wrong_workers))
    with pytest.raises(RuntimeError, match="spawned input-worker"):
        summarize(tmp_path, devices, 400, frozen, temporal=temporal)
    train_path.write_text(original_train)
    if temporal:
        for altered in (
            {"temporal_training_samples": []},
            {"architecture": architecture_facts(model_factory())},
        ):
            train_path.write_text(json.dumps(json.loads(original_train) | altered))
            with pytest.raises(RuntimeError, match="temporal training losses|architecture"):
                summarize(tmp_path, devices, 400, frozen, temporal=True)
        bad_temporal = json.loads(original_train)
        bad_temporal["temporal_training_samples"][0]["metrics"]["temporal_variance_loss"] = float(
            "nan"
        )
        train_path.write_text(json.dumps(bad_temporal))
        with pytest.raises(RuntimeError, match="nonfinite"):
            summarize(tmp_path, devices, 400, frozen, temporal=True)
        train_path.write_text(original_train)
    damaged = json.loads((tmp_path / "validation-rank-1.json").read_text())
    damaged["ema_updates"] = 399
    (tmp_path / "validation-rank-1.json").write_text(json.dumps(damaged))
    with pytest.raises(RuntimeError, match="Rank reports disagree"):
        summarize(tmp_path, devices, 400, frozen, temporal=temporal)
    with pytest.raises(RuntimeError, match="nonfinite"):
        finite_metrics([{"val/loss": float("nan")}])


def test_scratch_is_unique_and_outside_artifact_tree_even_if_cleanup_never_runs(tmp_path):
    from scripts.benchmark_streaming_gpu import scratch_directory

    output = tmp_path / "reports"
    first = scratch_directory(tmp_path / "scratch", output)
    second = scratch_directory(tmp_path / "scratch", output)
    assert first != second
    assert first.is_dir() and second.is_dir()
    (first / "large-cache.tar").write_bytes(b"partial shard")
    assert not output.exists()
    for forbidden in (output, output / "nested", Path("/output/cache")):
        with pytest.raises(ValueError, match="outside /output"):
            scratch_directory(forbidden, output)


@pytest.mark.parametrize("seconds", [30, 600, 900.25])
def test_durable_probe_uses_same_transfer_timeout_for_nccl_and_lightning(tmp_path, seconds):
    args = durable_args(tmp_path, checkpoint_transfer_timeout_seconds=seconds)
    cfg = make_config(
        args, {"path": "/tmp/frozen.json", "frozen_index_sha256": "a" * 64}, tmp_path / "cache"
    )
    strategy = instantiate(cfg.trainer.strategy)
    assert strategy._timeout == timedelta(seconds=collective_timeout_seconds(args))
    assert strategy._timeout.total_seconds() >= seconds + 120


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_prefix": None},
        {"exclusive_checkpoint_writer": False},
        {"devices": 2},
        {"workers": 2},
        {"checkpoint_transfer_timeout_seconds": 0},
        {"checkpoint_transfer_timeout_seconds": float("nan")},
    ],
)
def test_durable_probe_rejects_incomplete_or_weaker_configuration(tmp_path, changes):
    with pytest.raises(ValueError):
        durable_checkpoint_options(durable_args(tmp_path, **changes))


def test_checkpoint_probe_refuses_a_foreign_prefix(tmp_path, monkeypatch):
    client = FakeS3()
    monkeypatch.setattr(
        "scripts.benchmark_streaming_gpu.checkpoint_client", lambda endpoint: client
    )
    args = durable_args(tmp_path)
    client.objects[args.checkpoint_prefix + "/foreign"] = b"preserve"
    with pytest.raises(ValueError, match="unowned/nonempty"):
        open_checkpoint_repository(args, {"frozen_index_sha256": "a" * 64})
    assert client.objects[args.checkpoint_prefix + "/foreign"] == b"preserve"


@pytest.mark.parametrize("failure", ["readback", "download", "timeout"])
def test_durable_probe_failure_never_returns_success_evidence(tmp_path, monkeypatch, failure):
    import io

    class DamagedS3(FakeS3):
        checkpoint_reads = 0

        def get_object(self, **kwargs):
            result = super().get_object(**kwargs)
            if kwargs["Key"].endswith(".ckpt"):
                self.checkpoint_reads += 1
                corrupt_at = {"readback": 1, "download": 2}.get(failure)
                if self.checkpoint_reads == corrupt_at:
                    payload = result["Body"].read()
                    result["Body"] = io.BytesIO(bytes([payload[0] ^ 1]) + payload[1:])
            return result

    client = DamagedS3()
    monkeypatch.setattr(
        "scripts.benchmark_streaming_gpu.checkpoint_client", lambda endpoint: client
    )
    repository = open_checkpoint_repository(
        durable_args(tmp_path), {"frozen_index_sha256": "a" * 64}
    )
    path = tmp_path / "checkpoint.ckpt"
    path.write_bytes(b"checkpoint fixture: fails before deserialization")
    clock = [0]
    if failure == "timeout":
        monkeypatch.setattr("scripts.benchmark_streaming_gpu.time.monotonic", lambda: clock[0])
        publish = repository.publish

        def late_publication(*args, **kwargs):
            result = publish(*args, **kwargs)
            clock[0] = 11
            return result

        monkeypatch.setattr(repository, "publish", late_publication)
    with pytest.raises((ValueError, TimeoutError), match="checksum|SHA256|transfer budget"):
        durable_checkpoint_roundtrip(repository, path, None, None, 2, 10)
    assert not path.with_name("downloaded-checkpoint.ckpt").exists()
    assert not path.with_name("downloaded-checkpoint.partial").exists()
    assert not (tmp_path / "streaming-benchmark-success.json").exists()

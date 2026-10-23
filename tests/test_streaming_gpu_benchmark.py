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
    probe_variant,
    require_architecture,
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


@pytest.mark.parametrize(("temporal", "nat"), [(False, False), (True, False), (True, True)])
def test_full_checkpoint_after_validation_preserves_training_and_data_state(
    tmp_path, monkeypatch, temporal, nat
):
    from test_hierarchical_dynamics import model_factory

    class FixtureData(L.LightningDataModule):
        def train_dataloader(self):
            items = [
                {"waveform": torch.randn(2, 16160 * 9), "masked_crops": torch.randn(8, 2, 32160)}
                if nat
                else {"waveform": torch.randn(512 * 9)}
                for _ in range(4)
            ]
            return DataLoader(items, batch_size=2)

        val_dataloader = train_dataloader

        def state_dict(self):
            return {
                "index_sha256": "a" * 64,
                "stream_position_checkpointed": False,
                "naturalistic_sha256": "b" * 64 if nat else None,
            }

    model_options = {}
    if nat:
        from test_nat_backbone import nat_config

        from src.models.components.wavjepa_backbone import WavJEPABackbone

        model_options = {
            "backbone": WavJEPABackbone(
                nat_config(
                    window_samples=16160,
                    conv_layers=[[8, 80, 80], [8, 2, 2]],
                    encoder_layers=1,
                    decoder_layers=1,
                    average_top_k_layers=1,
                    compact_masked_tokens=True,
                    use_gradient_checkpointing=True,
                )
            ),
            "masked_crops_per_recording": 8,
            "masked_crop_samples": 32160,
        }
    steps = 51 if temporal and not nat else 2
    model = model_factory(
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
        aggregation_windows=[1, 2, 2] if temporal else [1, 1, 1],
        temporal_variance_weight=0.1 if temporal else 0.0,
        training_metric_interval=50 if steps == 51 else 1,
        **model_options,
    )
    data = FixtureData()
    # Exercise the real callback ordering/metric names on CPU. Only CUDA timing
    # and the separately tested worker inventory are substituted here.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    if nat:
        # This test exercises actual CNN updates; remote-cache inventory has a
        # separate FakeS3/eviction test below instead of renting cloud storage.
        monkeypatch.setattr("scripts.benchmark_streaming_gpu._nat_asset_catalog", lambda *args: {})
        inventories = iter(
            [
                {"contracts": 1, "brir": {"initial-brir": 10}, "noise": {"initial-noise": 20}},
                {"contracts": 1, "brir": {"new-brir": 10}, "noise": {"new-noise": 20}},
            ]
        )
        monkeypatch.setattr(
            "scripts.benchmark_streaming_gpu._nat_cache_inventory", lambda *args: next(inventories)
        )
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
        nat=nat,
        naturalistic_sha256="b" * 64 if nat else None,
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
        assert [sample["optimizer_step"] for sample in timing.temporal_samples] == (
            [1, 50] if steps == 51 else [1, 2]
        )
    rank_report = json.loads((tmp_path / "rank-0.json").read_text())
    assert rank_report["architecture"] == architecture_facts(model)
    assert rank_report["temporal_training_samples"] == timing.temporal_samples
    if nat:
        from scripts.benchmark_streaming_gpu import _require_nat_training_evidence

        _require_nat_training_evidence(rank_report, "b" * 64)
        assert all(row["changed_parameter_tensors"] > 0 for row in rank_report["ear_cnn_updates"])
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
        durable_args(
            tmp_path, nat=nat, naturalistic_manifest=tmp_path / "assets.json" if nat else None
        ),
        {"frozen_index_sha256": "a" * 64},
    )
    report = durable_checkpoint_roundtrip(repository, path, model, data, steps, 600)
    assert report["complete_remote_readback_verified"] is True
    assert report["download_sha256_verified"] is True
    assert report["strict_tensor_and_metadata_roundtrip"] is True
    assert report["architecture"] == geometry
    assert report["receipt"]["checkpoint"]["step"] == steps
    assert report["receipt"]["owner"]["global_batch"] == (16 if nat else 64)
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
                    "global_batch": 64,
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
                "owner": {"index_sha256": frozen["frozen_index_sha256"], "global_batch": 64},
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


def test_nat_probe_config_keeps_binaural_crops_capacity_flags_and_global_batch(tmp_path):
    args = durable_args(tmp_path, nat=True, naturalistic_manifest=tmp_path / "assets.json")
    cfg = make_config(
        args,
        {
            "path": "/tmp/frozen.json",
            "frozen_index_sha256": "a" * 64,
            "naturalistic_manifest_sha256": "b" * 64,
        },
        tmp_path / "cache",
    )
    validate_config(cfg)
    assert cfg.trainer.devices == 4 and cfg.data.batch_size == 4
    assert cfg.data.batch_size * cfg.trainer.devices * cfg.data.masked_crops_per_recording == 128
    assert cfg.data.naturalistic_manifest == str((tmp_path / "assets.json").resolve())
    assert cfg.data.masked_crops_per_recording == cfg.model.masked_crops_per_recording == 8
    assert cfg.data.masked_crop_samples == cfg.model.masked_crop_samples == 32160
    assert cfg.model.backbone.config.input_channels == 2
    assert cfg.model.backbone.config.use_gradient_checkpointing is True
    assert cfg.model.backbone.config.compact_masked_tokens is True
    assert cfg.callbacks.timing.temporal is cfg.callbacks.timing.nat is True
    assert cfg.callbacks.timing.naturalistic_sha256 == "b" * 64
    assert cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.data.num_workers == 1 and cfg.trainer.use_distributed_sampler is False


@pytest.mark.parametrize(
    "changes",
    [
        {"nat": True},
        {"nat": True, "devices": 2, "naturalistic_manifest": Path("assets.json")},
        {"nat": False, "naturalistic_manifest": Path("assets.json")},
    ],
)
def test_nat_probe_rejects_missing_assets_two_ranks_or_mono_fallback(tmp_path, changes):
    with pytest.raises(ValueError, match="Nat requires|mono fallback"):
        probe_variant(durable_args(tmp_path, **changes))


def test_nat_checkpoint_prefix_records_16_recordings_and_rejects_mono_adoption(
    tmp_path, monkeypatch
):
    client = FakeS3()
    monkeypatch.setattr(
        "scripts.benchmark_streaming_gpu.checkpoint_client", lambda endpoint: client
    )
    args = durable_args(tmp_path, nat=True, naturalistic_manifest=tmp_path / "assets.json")
    repository = open_checkpoint_repository(args, {"frozen_index_sha256": "a" * 64})
    assert repository.owner["global_batch"] == 16
    with pytest.raises(ValueError, match="different run"):
        open_checkpoint_repository(durable_args(tmp_path), {"frozen_index_sha256": "a" * 64})


def test_nat_architecture_receipt_cannot_be_accepted_as_mono_or_without_eight_crops():
    from test_hierarchical_dynamics import model_factory
    from test_nat_backbone import nat_config

    from src.models.components.wavjepa_backbone import WavJEPABackbone

    model = model_factory(
        backbone=WavJEPABackbone(nat_config()),
        masked_crops_per_recording=8,
        masked_crop_samples=32160,
        aggregation_windows=[1, 2, 2],
        temporal_variance_weight=0.1,
    )
    facts = architecture_facts(model)
    require_architecture(facts, temporal=True, nat=True)
    for field in ("input_channels", "masked_crops_per_recording", "masked_crop_samples"):
        incomplete = {key: value for key, value in facts.items() if key != field}
        with pytest.raises(RuntimeError, match="architecture"):
            require_architecture(incomplete, temporal=True, nat=True)
    for changed in (facts, facts | {"input_channels": 1}):
        with pytest.raises(RuntimeError, match="architecture"):
            require_architecture(changed, temporal=True)
    with pytest.raises(RuntimeError, match="architecture"):
        require_architecture(facts | {"masked_crops_per_recording": 1}, temporal=True, nat=True)


@pytest.mark.parametrize("batch", [0, -1, True, 1.5])
def test_checkpoint_repository_rejects_invalid_global_recording_batch(batch):
    from scripts.segment_checkpoint import CheckpointRepository

    with pytest.raises(ValueError, match="global recording batch"):
        CheckpointRepository(
            FakeS3(),
            bucket="test",
            prefix="probes/probe-run-001",
            run_id="probe-run-001",
            index_sha256="a" * 64,
            check=lambda: None,
            global_batch=batch,
        )


def test_nat_summary_requires_128_crops_pinned_manifest_and_correct_checkpoint_population(tmp_path):
    from copy import deepcopy

    from test_training_segments import streaming_report

    report = streaming_report()
    report["architecture"].update(
        input_channels=2, masked_crops_per_recording=8, masked_crop_samples=32160
    )
    frozen = report["frozen_index"] | {"naturalistic_manifest_sha256": "b" * 64}
    for rank in report["ranks"]:
        rank.update(
            global_batch=16,
            global_masked_crops=128,
            naturalistic_manifest_sha256="b" * 64,
            mean_seconds_per_update=1.5,
            nat_asset_streaming={
                "manifest_sha256": "b" * 64,
                "train_end": {
                    "remote_cache_contracts": 1,
                    "brir": {"verified_files": 2, "verified_bytes": 100},
                    "noise": {"verified_files": 2, "verified_bytes": 200},
                },
                "new_verified_brir_objects_after_warmup": 1,
                "new_verified_noise_objects_after_warmup": 1,
            },
            ear_cnn_updates=[
                {
                    "ear": ear,
                    "independent_parameters": True,
                    "parameter_tensors": 2,
                    "changed_parameter_tensors": 2,
                    "all_parameters_finite": True,
                }
                for ear in (0, 1)
            ],
        )
    for val in report["validation"]:
        val["naturalistic_manifest_sha256"] = "b" * 64
    report["checkpoint"]["durable_s3"]["receipt"]["owner"]["global_batch"] = 16

    def write(value):
        for rank, row in enumerate(value["ranks"]):
            (tmp_path / f"rank-{rank}.json").write_text(json.dumps(row))
        for rank, row in enumerate(value["validation"]):
            (tmp_path / f"validation-rank-{rank}.json").write_text(json.dumps(row))
        (tmp_path / "checkpoint.json").write_text(json.dumps(value["checkpoint"]))

    write(report)
    result = summarize(tmp_path, 4, 400, frozen, require_durable_checkpoint=True, nat=True)
    assert result["model_variant"] == "nat_temporal_aggregation"
    assert result["global_recording_batch"] == 16 and result["global_masked_crops"] == 128
    for damage in (
        "batch",
        "crops",
        "manifest",
        "validation",
        "owner",
        "architecture",
        "no_nat_cache",
        "no_brir_refill",
        "no_noise_refill",
        "frozen_ear",
        "nonfinite_ear",
    ):
        changed = deepcopy(report)
        if damage == "batch":
            changed["ranks"][1]["global_batch"] = 64
        elif damage == "crops":
            changed["ranks"][2]["global_masked_crops"] = 16
        elif damage == "manifest":
            changed["validation"][3]["naturalistic_manifest_sha256"] = "c" * 64
        elif damage == "validation":
            changed["validation"][0]["clips"] = 16
        elif damage == "owner":
            changed["checkpoint"]["durable_s3"]["receipt"]["owner"]["global_batch"] = 64
        elif damage == "architecture":
            changed["ranks"][0]["architecture"]["input_channels"] = 1
        elif damage == "no_nat_cache":
            changed["ranks"][1].pop("nat_asset_streaming")
        elif damage in ("no_brir_refill", "no_noise_refill"):
            kind = "brir" if damage == "no_brir_refill" else "noise"
            changed["ranks"][2]["nat_asset_streaming"][
                f"new_verified_{kind}_objects_after_warmup"
            ] = 0
        elif damage == "frozen_ear":
            changed["ranks"][3]["ear_cnn_updates"][1]["changed_parameter_tensors"] = 0
        elif damage == "nonfinite_ear":
            changed["ranks"][0]["ear_cnn_updates"][0]["all_parameters_finite"] = False
        write(changed)
        with pytest.raises(RuntimeError):
            summarize(tmp_path, 4, 400, frozen, require_durable_checkpoint=True, nat=True)
    write(report)
    with pytest.raises(RuntimeError):
        summarize(tmp_path, 4, 400, frozen, require_durable_checkpoint=True, temporal=True)


@pytest.mark.parametrize("global_batch", [16, 64])
def test_frozen_validation_coverage_uses_actual_recording_batch(
    tmp_path, monkeypatch, global_batch
):
    from scripts.benchmark_streaming_gpu import freeze_index

    index = {"status": "complete", "verified_sources": 889, "split_counts": {"val": 145}}
    monkeypatch.setenv("HWAVJEPA_INDEX_BUCKET", "fixture-bucket")
    monkeypatch.setenv("HWAVJEPA_INDEX_KEY", "index.json")
    monkeypatch.setattr(
        "src.data.sharded_audio_datamodule._read_index", lambda *args: (index, "a" * 64)
    )
    monkeypatch.setattr(
        "src.data.sharded_audio_datamodule._validate_index",
        lambda *args, **kwargs: {
            "train": [{}] * 4,
            "val": [{"splits": {"val": {"clips": 145}}}],
        },
    )
    frozen = freeze_index(
        tmp_path, "https://storage.example.invalid", devices=4, workers=1, global_batch=global_batch
    )
    assert frozen["validation_used_clips"] == (144 if global_batch == 16 else 128)
    assert frozen["validation_excluded_tail_clips"] == (1 if global_batch == 16 else 17)


@pytest.mark.parametrize(("nat", "steps"), [(False, 400), (True, 600)])
def test_cli_nat_default_exercises_more_than_one_training_shard(tmp_path, monkeypatch, nat, steps):
    from scripts.benchmark_streaming_gpu import main

    calls = []
    monkeypatch.setattr("scripts.benchmark_streaming_gpu.run", lambda args: calls.append(args))
    arguments = ["--devices", "4", "--output", str(tmp_path / "probe")]
    if nat:
        arguments += ["--nat", "--naturalistic-manifest", str(tmp_path / "assets.json")]
    main(arguments)
    assert calls[0].steps == steps
    assert calls[0].nat is nat


def test_nat_inventory_proves_remote_brir_and_noise_refills_despite_eviction(tmp_path, monkeypatch):
    import hashlib

    from test_naturalistic_audio import _assets, _remote_assets

    from scripts.benchmark_streaming_gpu import (
        _nat_asset_catalog,
        _nat_cache_inventory,
        _nat_streaming_facts,
    )
    from src.data.naturalistic_audio import NaturalisticAudio

    manifest_path, manifest = _assets(tmp_path)
    store = _remote_assets(manifest_path, manifest)
    payload = bytearray(store.objects["nat/whamr.wav"])
    payload[-1] ^= 1  # A second valid PCM noise asset with a different digest.
    store.objects["nat/whamr2.wav"] = bytes(payload)
    manifest["noise_files"].append("whamr2.wav")
    manifest["assets"]["whamr2.wav"] = {
        "key": "nat/whamr2.wav",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    catalog = _nat_asset_catalog(manifest_path, manifest_sha)
    monkeypatch.setattr("src.data.sharded_audio_datamodule._s3_client", lambda endpoint: store)
    root = tmp_path / "cache"
    cache = root / "train/rank-2/worker-0/nat"
    budget = len(payload) + manifest["assets"]["source.npy"]["bytes"] + 1
    renderer = NaturalisticAudio(manifest_path, cache_dir=cache, cache_max_gib=budget / 1024**3)
    renderer._materialize((tmp_path / "source.npy").resolve())
    renderer._materialize((tmp_path / "whamr.wav").resolve())
    warm = _nat_cache_inventory(root, 2, catalog)
    renderer._materialize((tmp_path / "noise0.npy").resolve())
    renderer._materialize((tmp_path / "whamr2.wav").resolve())
    final = _nat_cache_inventory(root, 2, catalog)
    facts = _nat_streaming_facts(warm, final, manifest_sha)
    assert warm["contracts"] == final["contracts"] == 1
    assert len(final["brir"]) == len(warm["brir"]) == 1
    assert len(final["noise"]) == len(warm["noise"]) == 1
    assert facts["new_verified_brir_objects_after_warmup"] == 1
    assert facts["new_verified_noise_objects_after_warmup"] == 1
    assert len(store.calls) == 4
    assert _nat_cache_inventory(root, 0, catalog)["contracts"] == 0
    assert all(
        set(value) == {"verified_files", "verified_bytes"}
        for value in (facts["train_end"]["brir"], facts["train_end"]["noise"])
    )
    with pytest.raises(RuntimeError, match="pinned manifest"):
        _nat_cache_inventory(root, 2, catalog | {"manifest_sha256": "c" * 64})
    next(cache.glob("*.tar")).write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="size differs"):
        _nat_cache_inventory(root, 2, catalog)


def test_nat_inventory_rejects_local_assets_as_remote_streaming_evidence(tmp_path):
    import hashlib

    from test_naturalistic_audio import _assets, _with_checksums

    from scripts.benchmark_streaming_gpu import _nat_asset_catalog

    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    with pytest.raises(RuntimeError, match="remote asset manifest"):
        _nat_asset_catalog(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_nat_ear_update_evidence_detects_frozen_nonfinite_and_shared_cnn_parameters():
    from test_hierarchical_dynamics import model_factory
    from test_nat_backbone import nat_config

    from scripts.benchmark_streaming_gpu import _ear_cnns, _ear_update_facts
    from src.models.components.wavjepa_backbone import WavJEPABackbone

    model = model_factory(backbone=WavJEPABackbone(nat_config()))
    initial = [[p.detach().clone() for p in cnn.parameters()] for cnn in _ear_cnns(model)]
    with torch.no_grad():
        next(model.backbone.model.extract_audio.cnns[0].parameters()).add_(0.1)
    facts = _ear_update_facts(model, initial)
    assert facts[0]["changed_parameter_tensors"] == 1
    assert facts[1]["changed_parameter_tensors"] == 0
    assert all(row["all_parameters_finite"] for row in facts)
    with torch.no_grad():
        next(model.backbone.model.extract_audio.cnns[1].parameters()).fill_(float("nan"))
    assert _ear_update_facts(model, initial)[1]["all_parameters_finite"] is False
    model.backbone.model.extract_audio.cnns[1] = model.backbone.model.extract_audio.cnns[0]
    with pytest.raises(RuntimeError, match="independent ear CNN"):
        _ear_cnns(model)


@pytest.mark.parametrize("temporal", [False, True])
def test_legacy_mono_architecture_defaults_only_absent_channel_and_crop_metadata(temporal):
    from test_hierarchical_dynamics import model_factory

    model = model_factory(
        aggregation_windows=[1, 2, 2] if temporal else [1, 1, 1],
        temporal_variance_weight=0.1 if temporal else 0.0,
    )
    legacy = {
        key: value
        for key, value in architecture_facts(model).items()
        if key not in {"input_channels", "masked_crops_per_recording", "masked_crop_samples"}
    }
    require_architecture(legacy, temporal=temporal)
    assert "input_channels" not in legacy  # Validation never mutates the saved receipt.
    for change in (
        {"input_channels": 2},
        {"masked_crops_per_recording": 8},
        {"masked_crop_samples": 32160},
        {"unexpected_architecture_field": True},
    ):
        with pytest.raises(RuntimeError, match="architecture"):
            require_architecture(legacy | change, temporal=temporal)
    with pytest.raises(RuntimeError, match="architecture"):
        require_architecture(legacy, temporal=temporal, nat=True)

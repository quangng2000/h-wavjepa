"""CPU checks for the GPU probe's real config, lifecycle and success evidence."""

import json
from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.benchmark_streaming_gpu import (
    finite_metrics,
    make_config,
    summarize,
    verify_checkpoint,
    worker_facts,
)
from src.utils.runtime import validate_config


def test_worker_evidence_reads_actual_loader_context():
    loader = DataLoader([1, 2], num_workers=1, multiprocessing_context="spawn")
    assert worker_facts(loader) == {
        "data_worker_start_method": "spawn",
        "workers_per_rank": 1,
    }
    with pytest.raises(RuntimeError, match="spawned input workers"):
        worker_facts(DataLoader([1, 2], num_workers=1))


@pytest.mark.parametrize("devices", [2, 4])
def test_streaming_gpu_config_preserves_global_batch_objective_and_full_schedule(tmp_path, devices):
    args = SimpleNamespace(
        output=tmp_path,
        endpoint="https://storage.givemeanode.com",
        devices=devices,
        workers=1,
        steps=400,
        warmup=25,
        compact=True,
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
    assert cfg.trainer.limit_val_batches == 0 and cfg.test is False


def test_full_checkpoint_after_validation_preserves_training_and_data_state(tmp_path):
    from test_hierarchical_dynamics import model_factory

    class FixtureData(L.LightningDataModule):
        def train_dataloader(self):
            return DataLoader([{"waveform": torch.randn(512 * 9)} for _ in range(4)], batch_size=2)

        val_dataloader = train_dataloader

        def state_dict(self):
            return {"index_sha256": "a" * 64, "stream_position_checkpointed": False}

    model = model_factory(
        lr_schedule="cosine",
        lr_total_steps=375000,
        lr_warmup_steps=100000,
        ema_anneal_end_step=100000,
        ema_end_decay=0.99999,
    )
    data = FixtureData()
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model, datamodule=data)
    trainer.limit_val_batches = 1.0
    metrics = finite_metrics(trainer.validate(model=model, datamodule=data, verbose=False))
    assert metrics["val/loss"] > 0
    path = tmp_path / "probe.ckpt"
    trainer.save_checkpoint(path, weights_only=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    verify_checkpoint(checkpoint, model, data, 2)
    checkpoint[type(data).__qualname__] = {"index_sha256": "wrong"}
    with pytest.raises(RuntimeError, match="data-index"):
        verify_checkpoint(checkpoint, model, data, 2)


def test_summary_requires_matching_all_rank_training_validation_and_index(tmp_path):
    frozen = {"frozen_index_sha256": "a" * 64, "validation_used_clips": 128}
    for rank in range(2):
        identity = {
            "rank": rank,
            "world_size": 2,
            "optimizer_steps": 400,
            "ema_updates": 400,
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
                }
            )
        )
        (tmp_path / f"validation-rank-{rank}.json").write_text(
            json.dumps({**identity, "clips": 64, "metrics": {"val/loss": 1.0}})
        )
    (tmp_path / "checkpoint.json").write_text(
        json.dumps(
            {
                "strict_tensor_and_metadata_roundtrip": True,
                "bytes": 1024,
            }
        )
    )
    result = summarize(tmp_path, 2, 400, frozen)
    assert result["seconds_per_update"] == 0.6
    train_path = tmp_path / "rank-1.json"
    original_train = train_path.read_text()
    no_refill = json.loads(original_train)
    no_refill["rank_cache_at_train_end"]["verified_files"] = 1
    train_path.write_text(json.dumps(no_refill))
    with pytest.raises(RuntimeError, match="additional verified shard downloads"):
        summarize(tmp_path, 2, 400, frozen)
    train_path.write_text(original_train)
    wrong_workers = json.loads(original_train)
    wrong_workers["data_worker_start_method"] = "fork"
    train_path.write_text(json.dumps(wrong_workers))
    with pytest.raises(RuntimeError, match="spawned input-worker"):
        summarize(tmp_path, 2, 400, frozen)
    train_path.write_text(original_train)
    damaged = json.loads((tmp_path / "validation-rank-1.json").read_text())
    damaged["ema_updates"] = 399
    (tmp_path / "validation-rank-1.json").write_text(json.dumps(damaged))
    with pytest.raises(RuntimeError, match="Rank reports disagree"):
        summarize(tmp_path, 2, 400, frozen)
    with pytest.raises(RuntimeError, match="nonfinite"):
        finite_metrics([{"val/loss": float("nan")}])

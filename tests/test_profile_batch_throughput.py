"""Bounded batch profiles preserve the recipe and cannot publish checkpoints."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from test_training_segments import plan as segment_plan

from scripts import profile_batch_throughput as profile
from src.utils.runtime import validate_config


def value(tmp_path):
    source = segment_plan(tmp_path) | {
        "base_data_seed": 42,
        "nat": False,
        "clean_binaural": True,
        "temporal": True,
        "global_batch": 16,
        "distributed": {"nnodes": 1, "nproc_per_node": 8, "node_rank": 0},
        "operator_stop": {
            "request_path": str(tmp_path / "stop.json"),
            "run_id": "run1",
            "segment_id": "segment1",
            "nonce": "a" * 32,
        },
    }
    return {
        "source_plan": source,
        "checkpoint": str(tmp_path / "step500.ckpt"),
        "checkpoint_sha256": "a" * 64,
        "resume_step": 500,
        "global_batch": 16,
        "warmup": 3,
        "measured": 20,
        "timeout_seconds": 360,
        "output": str(tmp_path / "profile"),
    }


def rank_report(request, rank):
    return {
        "rank": rank,
        "world_size": 8,
        "global_batch": request["global_batch"],
        "local_batch": request["global_batch"] // 8,
        "resume_step": request["resume_step"],
        "final_step": request["resume_step"] + request["warmup"] + request["measured"],
        "ema_updates": request["resume_step"] + request["warmup"] + request["measured"],
        "scheduler_step": request["resume_step"] + request["warmup"] + request["measured"],
        "optimizer_steps_match": True,
        "warmup_steps": request["warmup"],
        "measured_steps": request["measured"],
        "measured_seconds": 10 + rank,
        "median_seconds_per_update": 0.7,
        "p95_seconds_per_update": 0.8,
        "allocated_peak_gib": 14 + rank,
        "reserved_peak_gib": 15 + rank,
        "finite_parameters": True,
        "online_weights_changed": True,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_written": False,
    }


@pytest.mark.parametrize("batch", [16, 32, 64, 128])
def test_candidate_retains_full_recipe_and_only_timing_callback(tmp_path, batch):
    request = value(tmp_path) | {"global_batch": batch}
    original = deepcopy(request)
    cfg = profile.profile_config(request)
    validate_config(cfg)
    assert request == original
    assert cfg.trainer.devices == 8 and cfg.trainer.num_nodes == 1
    assert cfg.data.batch_size == batch // 8
    assert cfg.data.seed == 542 and cfg.seed == 42
    assert cfg.data.index_path == str(tmp_path / "work" / "frozen-index.json")
    assert cfg.data.cache_dir == str(tmp_path / "work" / "shard-cache")
    assert cfg.data.clean_binaural and cfg.data.naturalistic_manifest is None
    assert cfg.data.shard_prefetch == 1 and cfg.data.num_workers == 1
    assert cfg.data.masked_crops_per_recording == cfg.model.masked_crops_per_recording == 8
    assert cfg.data.masked_crop_samples == cfg.model.masked_crop_samples == 32160
    assert list(cfg.model.aggregation_windows) == [1, 2, 2]
    assert cfg.model.backbone.config.input_channels == 2
    assert cfg.model.backbone.config.use_gradient_checkpointing is False
    assert cfg.model.masked_weight == 1 and cfg.model.regularization_weight == 0.1
    assert cfg.model.temporal_variance_weight == 0.1
    assert cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.ckpt_path == request["checkpoint"]
    assert cfg.trainer.max_steps == 523
    assert cfg.trainer.limit_val_batches == cfg.trainer.num_sanity_val_steps == 0
    assert cfg.trainer.enable_checkpointing is False and cfg.logger is None
    assert list(cfg.callbacks) == ["timing"]
    assert cfg.callbacks.timing._target_ == "scripts.profile_batch_throughput.ProfileTiming"
    # Production callbacks and publication must never be instantiated by the profiler.
    callbacks = OmegaConf.to_container(cfg.callbacks, resolve=True)
    assert "_target_" not in callbacks["timing"]["profile"]["source_plan"]["operator_stop"]
    assert cfg.paths.output_dir == str(tmp_path / "profile" / "run")


@pytest.mark.parametrize(
    "change",
    [
        {"global_batch": 8},
        {"global_batch": True},
        {"warmup": 0},
        {"warmup": 11},
        {"measured": 51},
        {"timeout_seconds": 29},
        {"timeout_seconds": 601},
        {"resume_step": 0},
        {"resume_step": 375000},
        {"checkpoint_sha256": "a" * 63},
    ],
)
def test_rejects_unbounded_or_unpinned_profile(tmp_path, change):
    with pytest.raises(ValueError):
        profile.validate_profile(value(tmp_path) | change)


@pytest.mark.parametrize("change", [{"nat": True}, {"clean_binaural": False}, {"distributed": {}}])
def test_requires_exact_source_recipe_and_eight_rank_layout(tmp_path, change):
    request = value(tmp_path)
    request["source_plan"].update(change)
    with pytest.raises(ValueError):
        profile.validate_profile(request)


def test_summary_uses_slowest_rank_and_distinguishes_examples_from_updates(tmp_path):
    request = value(tmp_path) | {"global_batch": 64}
    reports = [rank_report(request, rank) for rank in range(8)]
    result = profile.summarize(request, reports)
    assert result["optimizer_updates_per_second"] == 20 / 17
    assert result["recordings_per_second"] == 64 * 20 / 17
    assert result["masked_crops_per_second"] == 8 * 64 * 20 / 17
    assert result["allocated_peak_gib"] == 21
    assert result["reserved_peak_gib"] == 22
    assert result["checkpoint_written"] is False and result["profile_only"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"rank": 1},
        {"local_batch": 3},
        {"final_step": 522},
        {"ema_updates": 522},
        {"scheduler_step": 522},
        {"optimizer_steps_match": False},
        {"warmup_steps": 2},
        {"checkpoint_sha256": "b" * 64},
        {"finite_parameters": False},
        {"online_weights_changed": False},
        {"checkpoint_written": True},
        {"measured_seconds": 0},
        {"measured_seconds": float("nan")},
        {"allocated_peak_gib": float("inf")},
    ],
)
def test_summary_requires_consistent_evidence_from_every_rank(tmp_path, change):
    request = value(tmp_path)
    reports = [rank_report(request, rank) for rank in range(8)]
    reports[0].update(change)
    with pytest.raises(ValueError):
        profile.summarize(request, reports)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("damage", [{"ema_updates": 500}, {"finite_parameters": False}])
def test_every_rank_must_report_advanced_ema_and_finite_weights(tmp_path, rank, damage):
    request = value(tmp_path)
    reports = [rank_report(request, current) for current in range(8)]
    reports[rank].update(damage)
    with pytest.raises(ValueError, match="rank evidence"):
        profile.summarize(request, reports)


def test_callback_proves_complete_resume_finite_updated_weights_and_measured_steps(
    tmp_path, monkeypatch
):
    request = value(tmp_path) | {"warmup": 1, "measured": 2}
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.model = torch.nn.Module()
    model.backbone.model.encoder = torch.nn.Linear(2, 2)
    model.ema_updates = 500
    trainer = SimpleNamespace(
        world_size=8,
        global_rank=0,
        global_step=500,
        optimizers=[SimpleNamespace(state={"restored": {"step": torch.tensor(500)}})],
        lr_scheduler_configs=[SimpleNamespace(scheduler=SimpleNamespace(last_epoch=500))],
    )
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2**31)
    monkeypatch.setattr(profile.dist, "barrier", lambda: None)
    ticks = iter([10.0, 10.4, 10.9])
    monkeypatch.setattr(profile.time, "perf_counter", lambda: next(ticks))
    callback = profile.ProfileTiming(request)
    callback.on_train_start(trainer, model)
    with torch.no_grad():
        model.backbone.model.encoder.weight.add_(0.1)
    for step in range(501, 504):
        trainer.global_step = model.ema_updates = step
        trainer.optimizers[0].state["restored"]["step"] = torch.tensor(step)
        trainer.lr_scheduler_configs[0].scheduler.last_epoch = step
        callback.on_train_batch_end(trainer, model, {}, None, step - 501)
    callback.on_train_end(trainer, model)
    report = json.loads((tmp_path / "profile" / "rank-0.json").read_text())
    assert report["measured_seconds"] == pytest.approx(0.9)
    assert report["measured_steps"] == 2
    assert report["allocated_peak_gib"] == 1
    assert report["finite_parameters"] and report["online_weights_changed"]
    assert report["ema_updates"] == report["scheduler_step"] == 503
    assert report["optimizer_steps_match"] is True
    assert report["checkpoint_written"] is False
    assert not list(tmp_path.rglob("*.ckpt"))


def test_supervisor_timeout_reaps_owned_workers_and_records_failure(tmp_path, monkeypatch):
    import psutil

    from scripts import train_segment

    request = value(tmp_path)
    checkpoint = tmp_path / "step500.ckpt"
    checkpoint.write_bytes(b"fixture")
    request["checkpoint_sha256"] = profile.file_sha256(checkpoint)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *args, **kwargs: process)
    child = SimpleNamespace(pid=123, create_time=lambda: 1)
    monkeypatch.setattr(psutil, "Process", lambda pid: child)
    monkeypatch.setattr(train_segment, "remember_descendants", lambda owned: None)
    cleanup = []
    monkeypatch.setattr(
        train_segment,
        "stop_process_tree",
        lambda proc, owned, **kwargs: cleanup.append((proc, dict(owned))),
    )
    ticks = iter([0, 361, 362])
    monkeypatch.setattr(profile.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError):
        profile.run_candidate(request)
    assert cleanup == [(process, {(123, 1): child})]
    report = json.loads((tmp_path / "profile" / "profile-result.json").read_text())
    assert report["status"] == "failed" and report["exception_type"] == "TimeoutError"
    assert report["checkpoint_written"] is False


def test_sha_mismatch_cannot_launch_a_profile(tmp_path, monkeypatch):
    request = value(tmp_path)
    (tmp_path / "step500.ckpt").write_bytes(b"changed")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(profile.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="pinned SHA256"):
        profile.run_candidate(request)


def test_older_plan_uses_owned_base_seed_without_guessing_from_resolved_seed(tmp_path):
    request = value(tmp_path)
    del request["source_plan"]["base_data_seed"]
    request["source_plan"]["repository_options"]["base_data_seed"] = 42
    assert profile.profile_config(request).data.seed == 542
    del request["source_plan"]["repository_options"]["base_data_seed"]
    with pytest.raises(ValueError, match="base data seed"):
        profile.validate_profile(request)


def test_source_owner_and_plan_base_seed_must_agree(tmp_path):
    request = value(tmp_path)
    request["source_plan"]["repository_options"]["base_data_seed"] = 43
    with pytest.raises(ValueError, match="base data seed"):
        profile.validate_profile(request)

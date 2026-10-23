"""An explicit batch change forks lineage while restoring unchanged full state."""

import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_multihost_segments import config_plan
from test_training_segments import CheckpointS3, clean_curriculum_state, repository

from scripts.segment_checkpoint import CheckpointRepository
from scripts.train_segment import (
    finish_clean_phase,
    import_batch_transition,
    import_clean_curriculum,
    make_config,
    resolved_global_batch,
    validate_launch_options,
)
from src.utils.runtime import file_sha256


def saved_source(tmp_path, global_batch=16):
    client = CheckpointS3()
    source = repository(client, clean_binaural=True, global_batch=global_batch)
    source.open(initialize=True)
    path = tmp_path / "source.ckpt"
    torch.save(clean_curriculum_state(), path)
    receipt = source.publish(path, 2, segment_id="source-segment", data_seed=42)
    return client, source, path, receipt


@pytest.mark.parametrize("new_batch", [32, 64, 128])
def test_batch_change_imports_exact_full_state_into_distinct_audited_owner(tmp_path, new_batch):
    client, source, path, old_receipt = saved_source(tmp_path)
    old_owner = dict(source.owner)
    destination = tmp_path / "restored.ckpt"
    step, receipt, transition = import_batch_transition(
        source,
        destination,
        expected_checkpoint_sha256=file_sha256(path),
        index_sha256="a" * 64,
        new_global_batch=new_batch,
        clean_binaural=True,
    )
    assert step == 2 and receipt == old_receipt
    assert destination.read_bytes() == path.read_bytes()  # Optimizer, EMA, LR and loops untouched.
    assert transition == {
        "source_run_id": source.run_id,
        "source_prefix": source.prefix,
        "source_checkpoint_sha256": file_sha256(path),
        "source_optimizer_step": 2,
        "old_global_batch": 16,
        "new_global_batch": new_batch,
    }
    changed = repository(
        client,
        run_id="batch-run-001",
        prefix="training/batch-run-001",
        clean_binaural=True,
        global_batch=new_batch,
        batch_transition=transition,
    )
    changed.open(initialize=True)
    assert changed.owner["batch_transition"] == transition
    assert source.owner == old_owner
    source.assert_owner()
    with pytest.raises(ValueError, match="different run"):
        repository(client, clean_binaural=True, global_batch=new_batch).open(initialize=False)
    owned = CheckpointRepository.from_owned_source(
        client,
        bucket=changed.store.bucket,
        prefix=changed.prefix,
        run_id=changed.run_id,
        index_sha256=changed.index_sha256,
        check=lambda: None,
    )
    assert owned.owner == changed.owner


@pytest.mark.parametrize("damage", ["sha", "same_batch", "phase", "manifest", "optimizer"])
def test_batch_transition_rejects_wrong_source_or_incomplete_full_state(tmp_path, damage):
    client, source, path, receipt = saved_source(tmp_path)
    if damage == "optimizer":
        state = clean_curriculum_state()
        state["optimizer_states"] = []
        torch.save(state, path)
        # Independently published corrupt full state still has a valid blob SHA.
        client, source, _, _ = saved_source(tmp_path, 16)
        state["global_step"] = state["state_dict"]["ema_updates"] = 4
        torch.save(state, path)
        receipt = source.publish(path, 4, segment_id="source-corrupt", data_seed=42)
    with pytest.raises(ValueError):
        import_batch_transition(
            source,
            tmp_path / "restored.ckpt",
            expected_checkpoint_sha256="f" * 64
            if damage == "sha"
            else receipt["checkpoint"]["sha256"],
            index_sha256="a" * 64,
            new_global_batch=16 if damage == "same_batch" else 64,
            clean_binaural=damage != "phase",
            naturalistic_manifest_sha256="b" * 64 if damage == "manifest" else None,
        )


@pytest.mark.parametrize("global_batch", [16, 32, 64, 128])
def test_eight_rank_batch_override_keeps_model_learning_rate_and_schedules(tmp_path, global_batch):
    base = config_plan(tmp_path) | {"distributed": {"nnodes": 1, "nproc_per_node": 8}}
    baseline = make_config(base)
    cfg = make_config(base | {"global_batch": global_batch})
    assert cfg.data.batch_size == global_batch // 8
    assert cfg.model == baseline.model
    assert cfg.trainer.max_steps == baseline.trainer.max_steps == 375000
    assert cfg.data.masked_crops_per_recording == 8
    assert cfg.data.num_workers == 1 and cfg.data.seed == baseline.data.seed


@pytest.mark.parametrize("value", [True, 0, 7, 8, 12, -8, 8.0])
def test_batch_override_must_be_integer_and_divisible_by_world(value):
    with pytest.raises(ValueError, match="divisible"):
        resolved_global_batch(SimpleNamespace(nnodes=1, nproc_per_node=8, global_batch_size=value))


def test_batch_transition_cannot_silently_reuse_owned_prefix_or_mix_phase_transition():
    valid = {
        "nnodes": 1,
        "nproc_per_node": 8,
        "clean_binaural": True,
        "global_batch_size": 64,
        "batch_source_prefix": "training/source-run",
        "batch_source_run_id": "source-run",
        "batch_source_checkpoint_sha256": "a" * 64,
        "checkpoint_prefix": "training/target-run",
        "resume": False,
    }
    validate_launch_options(SimpleNamespace(**valid))
    for change in (
        {"resume": True},
        {"global_batch_size": None},
        {"checkpoint_prefix": "training/source-run"},
        {"batch_source_checkpoint_sha256": None},
        {"curriculum_source_prefix": "training/other-run"},
    ):
        with pytest.raises(ValueError):
            validate_launch_options(SimpleNamespace(**(valid | change)))


def test_operator_pause_does_not_seal_prefix_and_later_nat_handoff_keeps_batch(tmp_path):
    client, source, path, receipt = saved_source(tmp_path, global_batch=64)
    result = {"stop_reason": "operator", "checkpoint": receipt}
    finish_clean_phase(source, result, tmp_path, time.time() + 60)
    assert result["ready_for_nat"] is False
    assert source.store.head(source.handoff_key) is None
    assert source.open(initialize=False) == receipt
    result["stop_reason"] = "bounded_segment"
    finish_clean_phase(source, result, tmp_path, time.time() + 60)
    assert result["ready_for_nat"] is True
    handoff = json.loads((tmp_path / "clean-handoff.json").read_text())
    assert handoff["global_batch"] == 64
    step, _, _ = import_clean_curriculum(
        source,
        tmp_path / "nat.ckpt",
        expected_checkpoint_sha256=file_sha256(path),
        index_sha256="a" * 64,
        target_manifest_sha256="b" * 64,
    )
    assert step == 2
    assert (tmp_path / "nat.ckpt").read_bytes() == path.read_bytes()


def test_config_rejects_a_single_recording_per_rank(tmp_path):
    plan = config_plan(tmp_path) | {
        "distributed": {"nnodes": 1, "nproc_per_node": 8},
        "global_batch": 8,
    }
    with pytest.raises(ValueError, match="two recordings"):
        make_config(plan)

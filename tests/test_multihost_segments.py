"""Four one-GPU hosts preserve one optimizer population and one checkpoint writer."""

import json
import time
from types import SimpleNamespace

import pytest

from scripts import train_segment as segment
from src.utils.runtime import validate_config


def arguments(**changes):
    values = {
        "nnodes": 4,
        "nproc_per_node": 1,
        "node_rank": 0,
        "master_addr": "100.64.0.2",
        "master_port": 29500,
        "resume": True,
        "resume_sha256": "b" * 64,
        "shared_started_at": time.time() - 2,
    }
    return SimpleNamespace(**(values | changes))


def config_plan(tmp_path):
    return {
        "work_dir": str(tmp_path / "work"),
        "output": str(tmp_path / "output"),
        "training_deadline": time.time() + 3600,
        "hard_deadline": time.time() + 5400,
        "repository_options": {"endpoint": "https://storage.example.invalid"},
        "segment_id": "segment-001",
        "budget": segment.SegmentBudget(2, 40, 15.12).report(),
        "checkpoint_every_steps": 5000,
        "data_endpoint": "https://storage.givemeanode.com",
        "resume_path": "/data/verified-resume.ckpt",
        "resume_step": 250,
        "resolved_data_seed": 292,
        "clean_binaural": True,
        "distributed": segment.launch_topology(arguments()),
    }


def test_four_hosts_keep_global_recording_and_masked_crop_batches(tmp_path):
    cfg = segment.make_config(config_plan(tmp_path))
    validate_config(cfg)
    assert cfg.trainer.devices == 1
    assert cfg.trainer.num_nodes == 4
    assert cfg.data.batch_size == 4
    assert cfg.data.batch_size * cfg.trainer.devices * cfg.trainer.num_nodes == 16
    assert cfg.data.masked_crops_per_recording == 8
    assert cfg.trainer.max_steps == cfg.model.lr_total_steps == 375000
    assert cfg.data.seed == 292


def test_persistent_cache_can_be_reused_outside_fresh_segment_work_directory(tmp_path):
    plan = config_plan(tmp_path)
    assert segment.make_config(plan).data.cache_dir == str(tmp_path / "work" / "shard-cache")
    plan["cache_dir"] = str(tmp_path / "persistent-audio-cache")
    cfg = segment.make_config(plan)
    assert cfg.data.cache_dir == str(tmp_path / "persistent-audio-cache")
    assert cfg.data.index_path == str(tmp_path / "work" / "frozen-index.json")


@pytest.mark.parametrize(("clean", "global_batch", "local_batch"), [(True, 16, 2), (False, 64, 8)])
def test_eight_local_gpus_preserve_original_global_batch(
    tmp_path, clean, global_batch, local_batch
):
    topology = segment.launch_topology(SimpleNamespace(nnodes=1, nproc_per_node=8))
    plan = config_plan(tmp_path) | {"distributed": topology, "clean_binaural": clean}
    cfg = segment.make_config(plan)
    validate_config(cfg)
    assert cfg.trainer.devices == 8 and cfg.trainer.num_nodes == 1
    assert cfg.data.batch_size == local_batch
    assert cfg.data.batch_size * cfg.trainer.devices == global_batch
    assert cfg.trainer.max_steps == cfg.model.lr_total_steps == 375000
    if clean:
        assert cfg.data.masked_crops_per_recording * global_batch == 128
    command = segment.torchrun_command(plan, tmp_path / "plan.json")
    assert "--standalone" in command and "--nproc_per_node=8" in command


def test_eight_gpus_do_not_reuse_four_gpu_acceptance_evidence():
    args = SimpleNamespace(nnodes=1, nproc_per_node=8)
    with pytest.raises(ValueError, match="topology.*waiver"):
        segment.streaming_acceptance(args)
    args.skip_streaming_acceptance = True
    args.streaming_acceptance_waiver_reason = (
        "User authorized eight-GPU training without another separate preflight."
    )
    receipt = segment.streaming_acceptance(args)
    assert receipt["status"] == "waived_by_user"
    assert receipt["hardware_acceptance_passed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"resume": False},
        {"resume_sha256": None},
        {"resume_sha256": "bad"},
        {"shared_started_at": None},
        {"shared_started_at": float("nan")},
        {"shared_started_at": time.time() + 86400},
        {"node_rank": 4},
        {"master_addr": "127.0.0.1"},
        {"master_addr": "8.8.8.8"},
        {"nproc_per_node": 4},
        {"curriculum_source_prefix": "new-phase"},
        {"execute_started": time.time() + 60},
    ],
)
def test_multihost_rejects_unpinned_or_inconsistent_launches(changes):
    with pytest.raises(ValueError):
        segment.launch_topology(arguments(**changes))


def test_existing_single_host_command_is_unchanged(tmp_path):
    topology = segment.launch_topology(SimpleNamespace())
    assert topology["nnodes"] == 1 and topology["nproc_per_node"] == 4
    command = segment.torchrun_command({}, tmp_path / "plan.json")
    assert "--standalone" in command
    assert "--nproc_per_node=4" in command
    assert "--local-addr=127.0.0.1" in command


def test_static_multihost_command_uses_private_master_and_unique_node_rank(tmp_path):
    command = segment.torchrun_command(
        {"distributed": segment.launch_topology(arguments(node_rank=3))}, tmp_path / "plan.json"
    )
    for value in (
        "--nnodes=4",
        "--nproc_per_node=1",
        "--node_rank=3",
        "--master_addr=100.64.0.2",
        "--master_port=29500",
        "--max_restarts=0",
    ):
        assert value in command
    assert "--standalone" not in command


def test_expired_shared_budget_never_starts_a_new_local_process(monkeypatch):
    monkeypatch.setattr(segment, "segment_budget", lambda args: SimpleNamespace(seconds=60))
    monkeypatch.setattr(segment.subprocess, "Popen", lambda *a, **k: pytest.fail("deadline reset"))
    with pytest.raises(TimeoutError, match="expired"):
        segment.supervise(SimpleNamespace(plan_only=False, shared_started_at=time.time() - 100), [])


@pytest.mark.parametrize("multihost", [True, False])
def test_nonzero_worker_never_writes_global_completion(tmp_path, monkeypatch, multihost):
    plan = config_plan(tmp_path)
    plan["distributed"] = (
        segment.launch_topology(arguments(node_rank=2))
        if multihost
        else segment.launch_topology(SimpleNamespace(nnodes=1, nproc_per_node=8))
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    receipt = {"checkpoint": {"step": 500, "sha256": "c" * 64}}
    trainer = SimpleNamespace(
        global_step=500,
        is_global_zero=False,
        checkpoint_callback=SimpleNamespace(last_receipt=receipt),
    )
    for name, value in {
        "WORLD_SIZE": "4" if multihost else "8",
        "RANK": "2" if multihost else "7",
        "LOCAL_RANK": "0" if multihost else "7",
        "LOCAL_WORLD_SIZE": "1" if multihost else "8",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(segment, "make_config", lambda plan: None)
    monkeypatch.setattr(
        "src.train.train",
        lambda cfg: ({}, {"trainer": trainer, "model": SimpleNamespace(ema_updates=500)}),
    )
    barriers = []
    monkeypatch.setattr("torch.distributed.barrier", lambda: barriers.append(1))
    segment.worker(path)
    output = tmp_path / "output"
    if multihost:
        follower = json.loads((output / "node-finished.json").read_text())
        assert follower["checkpoint_writer"] is False
        assert follower["node_rank"] == 2
        assert follower["checkpoint"] == receipt
    else:
        assert not (output / "node-finished.json").exists()
    assert not (output / "segment-finished.json").exists()
    assert len(barriers) == 2


def test_follower_finalization_reads_receipt_without_publishing_or_sealing(tmp_path, monkeypatch):
    receipt = {"checkpoint": {"step": 500, "sha256": "c" * 64}}
    result = {
        "node_finished": True,
        "checkpoint_writer": False,
        "node_rank": 1,
        "world_size": 4,
        "initial_optimizer_step": 250,
        "final_optimizer_step": 500,
        "checkpoint": receipt,
    }
    (tmp_path / "node-finished.json").write_text(json.dumps(result))
    store = SimpleNamespace(
        check=lambda: pytest.fail("expired setup deadline used for finalization")
    )

    def latest():
        store.check()
        return receipt

    repository = SimpleNamespace(store=store, latest=latest)
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("GMN_RESULT_PATH", str(result_path))
    outcome = segment.finish_follower(
        SimpleNamespace(output=tmp_path),
        repository,
        topology={"node_rank": 1},
        resume_step=250,
        hard_deadline=time.time() + 100,
    )
    assert outcome == result
    assert json.loads(result_path.read_text())["checkpoint_writer"] is False
    assert "ready_for_nat" not in outcome
    assert not (tmp_path / "clean-handoff.json").exists()
    with pytest.raises(TimeoutError):
        segment.finish_follower(
            SimpleNamespace(output=tmp_path),
            repository,
            topology={"node_rank": 1},
            resume_step=250,
            hard_deadline=time.time() - 1,
        )


def test_follower_rejects_unmatched_remote_final_checkpoint(tmp_path):
    result = {
        "node_finished": True,
        "checkpoint_writer": False,
        "node_rank": 1,
        "world_size": 4,
        "initial_optimizer_step": 250,
        "final_optimizer_step": 500,
        "checkpoint": {"checkpoint": {"step": 500}},
    }
    (tmp_path / "node-finished.json").write_text(json.dumps(result))
    repository = SimpleNamespace(
        store=SimpleNamespace(), latest=lambda: {"checkpoint": {"step": 250}}
    )
    with pytest.raises(RuntimeError, match="matching final"):
        segment.finish_follower(
            SimpleNamespace(output=tmp_path),
            repository,
            topology={"node_rank": 1},
            resume_step=250,
            hard_deadline=time.time() + 100,
        )


def test_follower_rejects_changed_resume_before_starting_any_rank(tmp_path, monkeypatch):
    args = arguments(node_rank=2)
    vars(args).update(
        nat=False,
        clean_binaural=True,
        temporal=True,
        plan_only=False,
        naturalistic_manifest=None,
        naturalistic_manifest_sha256=None,
        skip_streaming_acceptance=True,
        streaming_smoke_report=None,
        streaming_acceptance_waiver_reason="User authorized the existing production recipe.",
        segment_hours=2,
        budget_usd=40,
        maximum_rate_usd_per_hour=15.12,
        finalization_reserve_seconds=1800,
        transfer_timeout_seconds=600,
        exclusive_checkpoint_writer=True,
        acknowledge_stream_restart=True,
        execute_started=args.shared_started_at,
        output=tmp_path / "output",
        work_dir=tmp_path / "work",
        index=tmp_path / "index.json",
        index_sha256="a" * 64,
        data_endpoint="https://storage.givemeanode.com",
        checkpoint_endpoint="https://storage.givemeanode.com",
        checkpoint_bucket="checkpoints",
        checkpoint_prefix="training/example-run",
        run_id="example-run",
        base_data_seed=42,
    )
    accesses = []

    class ReadOnlyRepository:
        def __init__(self, *args, **kwargs):
            pass

        def open(self, *, initialize):
            assert initialize is False
            accesses.append("read_existing_owner")

        def download_latest(self, destination):
            return {"checkpoint": {"sha256": "c" * 64}}

    monkeypatch.setattr(segment, "CheckpointRepository", ReadOnlyRepository)
    monkeypatch.setattr(segment, "checkpoint_client", lambda endpoint: None)
    monkeypatch.setattr(segment, "freeze_complete_index", lambda *args: "a" * 64)
    monkeypatch.setattr(
        segment, "verify_resume_state", lambda *a, **k: pytest.fail("wrong state loaded")
    )
    monkeypatch.setattr(
        segment.subprocess, "Popen", lambda *a, **k: pytest.fail("wrong checkpoint trained")
    )
    with pytest.raises(ValueError, match="shared pinned SHA256"):
        segment.execute_segment(args)
    assert accesses == ["read_existing_owner"]

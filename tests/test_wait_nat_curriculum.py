"""CPU gate binds a completed clean writer to the verified Nat input corpus."""

import io
import json
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from test_start_nat_segment import handoff  # noqa: F401

from scripts import wait_nat_curriculum as gate
from scripts.start_nat_segment import segment_command, utc_timestamp


class Storage:
    def __init__(self, objects):
        self.objects = objects
        self.puts = []

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError(
                {"ResponseMetadata": {"HTTPStatusCode": 404}, "Error": {"Code": "NoSuchKey"}},
                "HeadObject",
            )
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, **options):
        return self.head_object(**options) | {"Body": io.BytesIO(self.objects[options["Key"]])}

    def put_object(self, *, Bucket, Key, Body, **_options):
        self.puts.append(Key)
        self.objects[Key] = Body


@pytest.fixture
def curriculum(handoff, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv("GMN_RESULT_PATH", raising=False)
    audio, nat, _, nat_options = handoff
    audio.head_object = lambda **kwargs: {"ContentLength": len(audio.objects[kwargs["Key"]])}
    run_id = "clean-curriculum-001"
    prefix = "training/" + run_id
    owner = gate.RECIPE | {
        "schema": 1,
        "owner": "h-wavjepa-training-segments",
        "run_id": run_id,
        "index_sha256": "a" * 64,
        "training_phase": "clean_binaural",
        "naturalistic_manifest_sha256": None,
        "base_data_seed": 42,
        "data_seed_policy": "base_plus_resume_step",
    }
    checkpoint = {
        "step": 123,
        "sha256": "b" * 64,
        "bytes": 10,
        "key": prefix + "/checkpoints/step_0000123_" + "b" * 64 + ".ckpt",
    }
    latest = {
        "owner": owner,
        "checkpoint": checkpoint,
        "resolved_data_seed": 42,
        "stream_positions_checkpointed": False,
    }
    clean = gate.RECIPE | {
        "schema_version": 1,
        "phase": "clean_binaural",
        "writer_finished": True,
        "ready_for_nat": True,
        "source_run_id": run_id,
        "source_prefix": prefix,
        "index_sha256": "a" * 64,
        "checkpoint": latest,
    }
    storage = Storage(
        {
            prefix + "/clean-handoff.json": json.dumps(clean).encode(),
            prefix + "/latest.json": json.dumps(latest).encode(),
            prefix + "/owner.json": json.dumps(owner).encode(),
            checkpoint["key"]: b"checkpoint",
        }
    )
    args = SimpleNamespace(
        clean_prefix=prefix,
        clean_run_id=run_id,
        index_sha256="a" * 64,
        checkpoint_bucket="checkpoints",
        curriculum_ready_key="curriculum/run/ready.json",
        nat_bucket=nat_options["bucket"],
        nat_endpoint=nat_options["endpoint"],
        nat_ready_key=nat_options["ready_key"],
        nat_manifest_path=nat_options["manifest_path"],
        nat_inventory=nat_options["inventory_path"],
        max_wait_seconds=10,
        poll_seconds=1,
        latest_start_utc=utc_timestamp("2099-01-01T00:00:00Z"),
        output=tmp_path / "output",
    )
    return args, audio, storage, clean, nat


def fetch(storage, args):
    return gate.fetch_clean_handoff(
        storage,
        bucket=args.checkpoint_bucket,
        prefix=args.clean_prefix,
        run_id=args.clean_run_id,
        index_sha256=args.index_sha256,
    )


def test_complete_gate_freezes_exact_optimizer_step_and_source_sha(curriculum):
    args, audio, storage, _, nat = curriculum
    result = gate.wait_for_curriculum(args, audio, storage)
    assert result["source_optimizer_step"] == 123
    assert result["source_checkpoint_sha256"] == "b" * 64
    assert result["ready_for_training"] is True
    assert json.loads((args.output / "curriculum-ready.json").read_text()) == result
    reread = gate.fetch_curriculum_handoff(
        storage,
        bucket=args.checkpoint_bucket,
        ready_key=args.curriculum_ready_key,
        source_prefix=args.clean_prefix,
        source_run_id=args.clean_run_id,
        index_sha256=args.index_sha256,
        nat_bucket=args.nat_bucket,
        nat_endpoint=args.nat_endpoint,
        nat_ready_key=args.nat_ready_key,
        nat_receipt=nat,
    )
    assert reread == result
    command = segment_command(
        [],
        manifest_path=args.nat_manifest_path,
        manifest_sha256=nat["manifest"]["sha256"],
        endpoint=args.nat_endpoint,
        curriculum=result,
    )
    assert command[-6:] == [
        "--curriculum-source-prefix",
        args.clean_prefix,
        "--curriculum-source-run-id",
        args.clean_run_id,
        "--curriculum-source-checkpoint-sha256",
        "b" * 64,
    ]


@pytest.mark.parametrize("damage", ["writer", "recipe", "index", "latest", "owner", "object"])
def test_invalid_clean_handoff_never_publishes_curriculum(curriculum, damage):
    args, audio, storage, clean, _ = curriculum
    if damage == "writer":
        clean["writer_finished"] = False
    elif damage == "recipe":
        clean["input_channels"] = 1
    elif damage == "index":
        clean["index_sha256"] = "c" * 64
    elif damage == "latest":
        latest = deepcopy(clean["checkpoint"])
        latest["resolved_data_seed"] = 43
        storage.objects[args.clean_prefix + "/latest.json"] = json.dumps(latest).encode()
    elif damage == "owner":
        storage.objects[args.clean_prefix + "/owner.json"] = b"{}"
    else:
        storage.objects[clean["checkpoint"]["checkpoint"]["key"]] = b"truncated"
    storage.objects[args.clean_prefix + "/clean-handoff.json"] = json.dumps(clean).encode()
    with pytest.raises(ValueError):
        gate.wait_for_curriculum(args, audio, storage)
    assert storage.puts == []


def test_cpu_wait_is_bounded_and_does_not_publish_on_timeout(curriculum):
    args, audio, storage, _, _ = curriculum
    clock = [0]

    def missing(**_kwargs):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")

    audio.head_object = missing
    with pytest.raises(TimeoutError, match="wait expired"):
        gate.wait_for_curriculum(
            args,
            audio,
            storage,
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        )
    assert clock[0] == args.max_wait_seconds
    assert storage.puts == []


def test_conflicting_curriculum_receipt_is_not_replaced(curriculum):
    args, audio, storage, _, _ = curriculum
    storage.objects[args.curriculum_ready_key] = b'{"different": true}'
    with pytest.raises(ValueError, match="refusing replacement"):
        gate.wait_for_curriculum(args, audio, storage)
    assert storage.objects[args.curriculum_ready_key] == b'{"different": true}'
    assert storage.puts == []


def test_gpu_bootstrap_rejects_source_advancing_after_gate(curriculum):
    args, audio, storage, clean, nat = curriculum
    gate.wait_for_curriculum(args, audio, storage)
    latest = clean["checkpoint"]
    latest["resolved_data_seed"] = 43
    storage.objects[args.clean_prefix + "/latest.json"] = json.dumps(latest).encode()
    with pytest.raises(ValueError, match="lineage"):
        gate.fetch_curriculum_handoff(
            storage,
            bucket=args.checkpoint_bucket,
            ready_key=args.curriculum_ready_key,
            source_prefix=args.clean_prefix,
            source_run_id=args.clean_run_id,
            index_sha256=args.index_sha256,
            nat_bucket=args.nat_bucket,
            nat_endpoint=args.nat_endpoint,
            nat_ready_key=args.nat_ready_key,
            nat_receipt=nat,
        )


def test_gate_import_does_not_load_gpu_training_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.wait_nat_curriculum; assert 'torch' not in sys.modules; assert 'lightning' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _change_source_batch(curriculum, batch, transition=None):
    args, _, storage, clean, _ = curriculum
    clean["global_batch"] = batch
    clean["checkpoint"]["owner"]["global_batch"] = batch
    if transition is not None:
        clean["checkpoint"]["owner"]["batch_transition"] = transition
    for suffix, value in (
        ("clean-handoff.json", clean),
        ("latest.json", clean["checkpoint"]),
        ("owner.json", clean["checkpoint"]["owner"]),
    ):
        storage.objects[args.clean_prefix + "/" + suffix] = json.dumps(value).encode()


def _batch_transition(batch):
    return {
        "source_run_id": "earlier-clean-001",
        "source_prefix": "training/earlier-clean-001",
        "source_checkpoint_sha256": "c" * 64,
        "source_optimizer_step": 100,
        "old_global_batch": 16,
        "new_global_batch": batch,
    }


@pytest.mark.parametrize("batch", [32, 64, 128])
def test_larger_batch_gate_preserves_recipe_change_into_nat_command(curriculum, batch):
    args, audio, storage, _, nat = curriculum
    transition = _batch_transition(batch)
    _change_source_batch(curriculum, batch, transition)
    result = gate.wait_for_curriculum(args, audio, storage)
    assert result["source_global_batch"] == batch
    assert result["source_batch_transition"] == transition
    reread = gate.fetch_curriculum_handoff(
        storage,
        bucket=args.checkpoint_bucket,
        ready_key=args.curriculum_ready_key,
        source_prefix=args.clean_prefix,
        source_run_id=args.clean_run_id,
        index_sha256=args.index_sha256,
        nat_bucket=args.nat_bucket,
        nat_endpoint=args.nat_endpoint,
        nat_ready_key=args.nat_ready_key,
        nat_receipt=nat,
    )
    assert reread == result
    command = segment_command(
        ["--nproc-per-node", "8"],
        manifest_path=args.nat_manifest_path,
        manifest_sha256=nat["manifest"]["sha256"],
        endpoint=args.nat_endpoint,
        curriculum=reread,
    )
    assert command[command.index("--global-batch-size") + 1] == str(batch)
    assert command[command.index("--curriculum-source-checkpoint-sha256") + 1] == "b" * 64
    assert "--nat" in command


@pytest.mark.parametrize("batch", [True, 0, 3, 16.0, "32"])
def test_clean_gate_rejects_invalid_source_population(curriculum, batch):
    _change_source_batch(curriculum, batch)
    args, audio, storage, _, _ = curriculum
    with pytest.raises(ValueError, match="global batch"):
        gate.wait_for_curriculum(args, audio, storage)
    assert not storage.puts


@pytest.mark.parametrize("damage", ["new_batch", "same_batch", "step", "sha", "prefix", "extra"])
def test_clean_gate_rejects_invalid_batch_transition_provenance(curriculum, damage):
    transition = _batch_transition(32)
    if damage == "new_batch":
        transition["new_global_batch"] = 64
    elif damage == "same_batch":
        transition["old_global_batch"] = 32
    elif damage == "step":
        transition["source_optimizer_step"] = 123
    elif damage == "sha":
        transition["source_checkpoint_sha256"] = "invalid"
    elif damage == "prefix":
        transition["source_prefix"] = "../training/earlier-clean-001"
    else:
        transition["unexpected"] = True
    _change_source_batch(curriculum, 32, transition)
    args, audio, storage, _, _ = curriculum
    with pytest.raises(ValueError, match="batch transition"):
        gate.wait_for_curriculum(args, audio, storage)
    assert not storage.puts


def test_gate_dynamic_batch_cannot_be_overridden_by_nat_segment_arguments(curriculum):
    args, audio, storage, _, nat = curriculum
    _change_source_batch(curriculum, 64, _batch_transition(64))
    result = gate.wait_for_curriculum(args, audio, storage)
    for arguments in (["--global-batch-size", "16"], ["--global-batch-size=16"]):
        with pytest.raises(ValueError, match="override"):
            segment_command(
                arguments,
                manifest_path=args.nat_manifest_path,
                manifest_sha256=nat["manifest"]["sha256"],
                endpoint=args.nat_endpoint,
                curriculum=result,
            )


def test_gpu_gate_rejects_population_or_provenance_changed_after_publication(curriculum):
    args, audio, storage, _, nat = curriculum
    _change_source_batch(curriculum, 64, _batch_transition(64))
    original = gate.wait_for_curriculum(args, audio, storage)
    for field, value in (("source_global_batch", 32), ("source_batch_transition", None)):
        changed = deepcopy(original)
        changed[field] = value
        storage.objects[args.curriculum_ready_key] = json.dumps(changed).encode()
        with pytest.raises(ValueError, match="changed after"):
            gate.fetch_curriculum_handoff(
                storage,
                bucket=args.checkpoint_bucket,
                ready_key=args.curriculum_ready_key,
                source_prefix=args.clean_prefix,
                source_run_id=args.clean_run_id,
                index_sha256=args.index_sha256,
                nat_bucket=args.nat_bucket,
                nat_endpoint=args.nat_endpoint,
                nat_ready_key=args.nat_ready_key,
                nat_receipt=nat,
            )


def test_legacy_batch16_gate_remains_compatible(curriculum):
    args, audio, storage, _, nat = curriculum
    receipt = gate.wait_for_curriculum(args, audio, storage)
    receipt.pop("source_global_batch")
    receipt.pop("source_batch_transition")
    storage.objects[args.curriculum_ready_key] = json.dumps(receipt).encode()
    actual = gate.fetch_curriculum_handoff(
        storage,
        bucket=args.checkpoint_bucket,
        ready_key=args.curriculum_ready_key,
        source_prefix=args.clean_prefix,
        source_run_id=args.clean_run_id,
        index_sha256=args.index_sha256,
        nat_bucket=args.nat_bucket,
        nat_endpoint=args.nat_endpoint,
        nat_ready_key=args.nat_ready_key,
        nat_receipt=nat,
    )
    command = segment_command(
        [],
        manifest_path=args.nat_manifest_path,
        manifest_sha256=nat["manifest"]["sha256"],
        endpoint=args.nat_endpoint,
        curriculum=actual,
    )
    assert command[command.index("--global-batch-size") + 1] == "16"

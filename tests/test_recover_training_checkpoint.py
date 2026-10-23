"""Pinned artifact recovery rejects unsafe inputs before any checkpoint write."""

import hashlib
import io
import json
import tarfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_training_segments import CheckpointS3, clean_curriculum_state

from scripts import recover_training_checkpoint as recovery
from scripts.segment_checkpoint import CheckpointRepository
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule


def clean_plan():
    options = {
        "bucket": "checkpoint-fixture",
        "prefix": "training/training-run-001",
        "run_id": "training-run-001",
        "index_sha256": "a" * 64,
        "base_data_seed": 42,
        "global_batch": 16,
        "naturalistic_manifest_sha256": None,
        "clean_binaural": True,
        "curriculum_source": None,
        "endpoint": "https://storage.example.invalid",
    }
    return {
        "repository_options": options,
        "clean_binaural": True,
        "nat": False,
        "temporal": True,
        "model_variant": "clean_binaural_temporal_aggregation",
        "global_batch": 16,
        "naturalistic_manifest_sha256": None,
        "curriculum_source": None,
        "index_sha256": "a" * 64,
        "base_data_seed": 42,
        "stream_positions_checkpointed": False,
        "data_seed_policy": "base_plus_resume_step",
        "resume_step": 0,
        "resolved_data_seed": 42,
        "segment_id": "clean-003",
    }


def full_clean_state():
    state = clean_curriculum_state()
    metadata = state[HierarchicalWavJEPAModule.CHECKPOINT_KEY]
    model = HierarchicalWavJEPAModule(
        backbone=WavJEPABackbone(metadata["backbone"]),
        optimizer=partial(torch.optim.AdamW, lr=0.001),
        **metadata["dynamics"],
    )
    configured = model.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    for _ in range(2):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    state["state_dict"] = model.state_dict()
    state["optimizer_states"] = [optimizer.state_dict()]
    state["lr_schedulers"] = [scheduler.state_dict()]
    return state


def checkpoint_bytes(state):
    output = io.BytesIO()
    torch.save(state, output)
    return output.getvalue()


def archive_bytes(*, plan=None, checkpoint=None, extra=()):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload in (
            ("./clean-003/segment-plan.json", json.dumps(plan or clean_plan()).encode()),
            ("./clean-003/checkpoints/step_0000002.ckpt", checkpoint or b"fixture-checkpoint"),
            ("./clean-003/metrics/ignored.csv", b"step,loss\n2,1\n"),
            *extra,
        ):
            if isinstance(payload, tarfile.TarInfo):
                archive.addfile(payload)
            else:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        alias = tarfile.TarInfo("./clean-003/checkpoints/last.ckpt")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "step_0000002.ckpt"
        archive.addfile(alias)
    return output.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self):
        return "https://artifact.example.invalid/object?signature=secret"


def use_download(monkeypatch, payload):
    def open_url(request, timeout):
        assert timeout == 60
        assert request.get_header("Accept-encoding") == "identity"
        return Response(payload)

    monkeypatch.setattr(recovery, "urlopen", open_url)
    monkeypatch.setenv(
        "HWAVJEPA_RECOVERY_ARTIFACT_URL",
        "https://artifact.example.invalid/object?signature=secret",
    )


def args_for(tmp_path, payload):
    return SimpleNamespace(
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_bytes=len(payload),
        checkpoint_step=2,
        timeout_seconds=60,
        output=tmp_path / "output",
        work_dir=tmp_path / "work",
    )


class ManagedFakeS3(CheckpointS3):
    def __init__(self):
        super().__init__()
        self.managed_uploads = 0

    def upload_file(self, Filename, Bucket, Key, Config, Callback):
        self.managed_uploads += 1
        payload = Path(Filename).read_bytes()
        Callback(len(payload))
        self.put_object(Bucket=Bucket, Key=Key, Body=payload, ContentLength=len(payload))


def owned_repository(client):
    options = dict(clean_plan()["repository_options"])
    options.pop("endpoint")
    repository = CheckpointRepository(client, **options, check=lambda: None)
    repository.open(initialize=True)
    return repository


def test_recovery_verifies_full_state_and_publishes_only_small_receipt(
    tmp_path, monkeypatch, capsys
):
    payload = archive_bytes(checkpoint=checkpoint_bytes(full_clean_state()))
    use_download(monkeypatch, payload)
    client = ManagedFakeS3()
    repository = owned_repository(client)
    monkeypatch.setattr(recovery, "checkpoint_client", lambda endpoint: client)
    args = args_for(tmp_path, payload)
    result = recovery.run_recovery(args)
    assert result["recovered"] is True
    assert result["full_training_state_verified"] is True
    assert result["model_state_strictly_loaded"] is True
    assert result["checkpoint"] == repository.latest()
    assert result["checkpoint"]["checkpoint"]["step"] == 2
    assert client.managed_uploads == 1
    assert not args.work_dir.exists()
    assert list(args.output.iterdir()) == [args.output / "recovery-success.json"]
    assert (args.output / "recovery-success.json").stat().st_size < 10000
    assert "signature" not in capsys.readouterr().out
    # Repeating recovery of the exact artifact verifies existing object bytes
    # and retains its receipt instead of uploading again or regressing latest.
    client.events.clear()
    args.output = tmp_path / "again"
    repeated = recovery.run_recovery(args)
    assert repeated["checkpoint"] == result["checkpoint"]
    assert client.managed_uploads == 1
    checkpoint_key = result["checkpoint"]["checkpoint"]["key"]
    assert ("get", checkpoint_key) in client.events


@pytest.mark.parametrize("bad_field", ["optimizer", "model"])
def test_incomplete_state_is_rejected_before_network(tmp_path, monkeypatch, bad_field):
    state = full_clean_state()
    if bad_field == "optimizer":
        state["optimizer_states"] = []
    else:
        state["state_dict"].pop(next(key for key in state["state_dict"] if key != "ema_updates"))
    payload = archive_bytes(checkpoint=checkpoint_bytes(state))
    use_download(monkeypatch, payload)
    monkeypatch.setattr(
        recovery, "checkpoint_client", lambda endpoint: pytest.fail("Must validate before network")
    )
    args = args_for(tmp_path, payload)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.run_recovery(args)
    assert not (args.output / "recovery-success.json").exists()
    assert not args.work_dir.exists()


def test_failed_publication_never_writes_success_or_leaks_url(tmp_path, monkeypatch, capsys):
    payload = archive_bytes(checkpoint=checkpoint_bytes(full_clean_state()))
    use_download(monkeypatch, payload)
    client = ManagedFakeS3()
    repository = owned_repository(client)
    client.fail_checkpoint = True
    monkeypatch.setattr(recovery, "checkpoint_client", lambda endpoint: client)
    args = args_for(tmp_path, payload)
    with pytest.raises(RuntimeError):
        recovery.run_recovery(args)
    assert repository.latest() is None
    assert not (args.output / "recovery-success.json").exists()
    failure = json.loads((args.output / "recovery-failed.json").read_text())
    assert failure["phase"] == "publish"
    assert failure["exception_chain"]
    assert "signature" not in capsys.readouterr().out
    assert not args.work_dir.exists()


@pytest.mark.parametrize("change", ["hash", "length", "deadline"])
def test_download_enforces_hash_size_and_deadline(tmp_path, monkeypatch, change):
    payload = b"pinned-small-artifact"
    use_download(monkeypatch, payload)
    digest = hashlib.sha256(payload).hexdigest()
    size = len(payload)

    def check():
        if change == "deadline":
            raise TimeoutError("test deadline")

    with pytest.raises((ValueError, TimeoutError)):
        recovery.download_artifact(
            "https://artifact.example.invalid/object",
            tmp_path / "archive.tar",
            expected_sha256="a" * 64 if change == "hash" else digest,
            expected_bytes=size - 1 if change == "length" else size,
            check=check,
        )
    assert not list(tmp_path.iterdir())


def test_extracts_only_plan_and_complete_regular_checkpoint(tmp_path):
    archive = tmp_path / "artifact.tar"
    archive.write_bytes(archive_bytes())
    work = tmp_path / "work"
    work.mkdir()
    plan, checkpoint, step, digest = recovery.extract_checkpoint(
        archive, work, expected_step=2, check=lambda: None
    )
    assert plan == clean_plan()
    assert step == 2
    assert checkpoint.read_bytes() == b"fixture-checkpoint"
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sorted(path.name for path in work.iterdir()) == [
        "segment-plan.json",
        "step_0000002.ckpt",
    ]
    assert not any(path.is_symlink() for path in work.iterdir())


def test_partial_checkpoint_cannot_be_selected(tmp_path):
    archive = tmp_path / "artifact.tar"
    with tarfile.open(archive, "w") as output:
        for name, payload in (
            ("./clean-003/segment-plan.json", json.dumps(clean_plan()).encode()),
            ("./clean-003/checkpoints/.step_0000002.ckpt.uuid.partial", b"unfinished"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="unambiguous complete checkpoint"):
        recovery.extract_checkpoint(archive, work, expected_step=2, check=lambda: None)
    assert not list(work.iterdir())


@pytest.mark.parametrize("success", [True, False])
def test_cli_writes_small_safe_platform_result(tmp_path, monkeypatch, capsys, success):
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("GMN_RESULT_PATH", str(result_path))

    def run(args):
        if not success:
            raise RuntimeError("https://private.invalid?signature=secret")
        return {"checkpoint": {"checkpoint": {"step": 250, "sha256": "a" * 64, "bytes": 123}}}

    monkeypatch.setattr(recovery, "run_recovery", run)
    argv = [
        "--artifact-sha256",
        "b" * 64,
        "--artifact-bytes",
        "1024",
        "--output",
        str(tmp_path / "out"),
        "--work-dir",
        str(tmp_path / "work"),
    ]
    if success:
        recovery.main(argv)
    else:
        with pytest.raises(SystemExit) as failure:
            recovery.main(argv)
        assert failure.value.code == 1
    result = json.loads(result_path.read_text())
    assert result["recovered"] is success
    assert result_path.stat().st_size < 1024
    assert "signature" not in result_path.read_text() + capsys.readouterr().out
    if success:
        assert result["checkpoint_step"] == 250
        assert result["complete_remote_readback_verified"] is True


@pytest.mark.parametrize(
    "unsafe", ["../escape", "/absolute", "clean-003/../../escape", "bad\\path"]
)
def test_rejects_traversal_before_extracting_any_files(tmp_path, unsafe):
    archive = tmp_path / "artifact.tar"
    archive.write_bytes(archive_bytes(extra=[(unsafe, b"no")]))
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        recovery.extract_checkpoint(archive, work, expected_step=None, check=lambda: None)
    assert not list(work.iterdir())


@pytest.mark.parametrize("kind", [tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.SYMTYPE])
def test_rejects_unsafe_links_and_special_members(tmp_path, kind):
    member = tarfile.TarInfo("clean-003/checkpoints/evil")
    member.type = kind
    member.linkname = "../../escape"
    archive = tmp_path / "artifact.tar"
    archive.write_bytes(archive_bytes(extra=[("ignored", member)]))
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError):
        recovery.extract_checkpoint(archive, work, expected_step=None, check=lambda: None)
    assert not list(work.iterdir())


def test_rejects_ambiguous_checkpoint_and_wrong_filename(tmp_path):
    for index, name in enumerate(("step_0000003.ckpt", "step_2.ckpt")):
        archive = tmp_path / f"artifact-{index}.tar"
        archive.write_bytes(archive_bytes(extra=[(f"clean-003/checkpoints/{name}", b"other")]))
        work = tmp_path / f"work-{index}"
        work.mkdir()
        with pytest.raises(ValueError):
            recovery.extract_checkpoint(archive, work, expected_step=None, check=lambda: None)


@pytest.mark.parametrize(
    "change",
    [
        {"clean_binaural": False},
        {"global_batch": 64},
        {"index_sha256": "b" * 64},
        {"resolved_data_seed": 43},
        {"resume_step": 2},
        {"nat": True},
    ],
)
def test_plan_lineage_mismatches_fail_locally(change):
    with pytest.raises(ValueError, match="plan"):
        recovery.repository_from_plan(deepcopy(clean_plan()) | change, lambda: None)

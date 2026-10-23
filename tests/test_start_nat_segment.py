"""Completed CPU asset handoff gates the production command, without a GPU probe."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scripts import start_nat_segment as bootstrap


class MetadataS3:
    def __init__(self, objects):
        self.objects = objects
        self.reads = []
        self.bodies = []
        self.declared_size = None

    def get_object(self, *, Bucket, Key):
        assert Bucket == "training-audio"
        self.reads.append(Key)
        payload = self.objects[Key]
        body = io.BytesIO(payload)
        self.bodies.append(body)
        return {
            "Body": body,
            "ContentLength": len(payload) if self.declared_size is None else self.declared_size,
        }


@pytest.fixture
def handoff(tmp_path):
    inventory = {"brir": {"repository": "official/scenes", "revision": "fixed-revision"}}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    manifest = {
        "schema_version": 1,
        "sample_rate": 16000,
        "split": "train",
        "bucket": "training-audio",
        "endpoint": "https://storage.example.com",
        "scenes": [{"id": str(index)} for index in range(70000)],
        "noise_files": [f"noise/{index}.wav" for index in range(20000)],
        "assets": {"one": {"key": "nat/assets/one"}},
        "provenance": {
            "coverage": "full",
            "remote_verified": True,
            "brir_repository": inventory["brir"]["repository"],
            "brir_revision": inventory["brir"]["revision"],
        },
    }
    payload = json.dumps(manifest).encode()
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "ready_for_training": True,
        "scene_count": 70000,
        "noise_count": 20000,
        "completed_components": 71,
        "expected_components": 71,
        "bucket": manifest["bucket"],
        "endpoint": manifest["endpoint"],
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "brir_repository": inventory["brir"]["repository"],
        "brir_revision": inventory["brir"]["revision"],
        "manifest": {
            "key": "nat/full/manifest.json",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    client = MetadataS3(
        {"nat/full/ready.json": json.dumps(receipt).encode(), receipt["manifest"]["key"]: payload}
    )
    options = {
        "bucket": receipt["bucket"],
        "endpoint": receipt["endpoint"],
        "ready_key": "nat/full/ready.json",
        "manifest_path": tmp_path / "scratch/manifest.json",
        "inventory_path": inventory_path,
    }
    return client, receipt, manifest, options


def test_complete_handoff_freezes_identical_bytes_and_never_imports_torch(handoff):
    client, receipt, _, options = handoff
    assert bootstrap.fetch_ready_manifest(client, **options) == receipt
    path = options["manifest_path"]
    assert path.read_bytes() == client.objects[receipt["manifest"]["key"]]
    assert json.loads(path.with_name(path.name + ".ready.json").read_text()) == receipt
    assert all(body.closed for body in client.bodies)
    # Test process imports Torch elsewhere; a fresh help invocation proves the
    # bootstrap itself can be loaded before the training stack is present.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.start_nat_segment; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "partial"),
        ("ready_for_training", False),
        ("ready_for_training", 1),
        ("scene_count", 69999),
        ("noise_count", 19999),
        ("completed_components", 70),
        ("inventory_sha256", "b" * 64),
        ("brir_revision", "other"),
        ("bucket", "other-bucket"),
    ],
)
def test_incomplete_or_different_inventory_stops_before_manifest_download(handoff, field, value):
    client, receipt, _, options = handoff
    receipt[field] = value
    client.objects[options["ready_key"]] = json.dumps(receipt).encode()
    with pytest.raises(ValueError, match="incomplete|inventory"):
        bootstrap.fetch_ready_manifest(client, **options)
    assert client.reads == [options["ready_key"]]
    assert not options["manifest_path"].exists()


@pytest.mark.parametrize("damage", ["digest", "size", "truncated", "oversized", "prefix"])
def test_changed_or_out_of_bounds_metadata_never_becomes_training_input(handoff, damage):
    client, receipt, _, options = handoff
    descriptor = receipt["manifest"]
    if damage == "digest":
        client.objects[descriptor["key"]] = client.objects[descriptor["key"]].replace(
            b'"train"', b'"other"', 1
        )
    elif damage == "size":
        descriptor["bytes"] += 1
    elif damage == "truncated":
        client.declared_size = len(client.objects[options["ready_key"]]) + 1
    elif damage == "oversized":
        descriptor["bytes"] = bootstrap.MAX_MANIFEST_BYTES + 1
    else:
        descriptor["key"] = "unrelated/manifest.json"
    client.objects[options["ready_key"]] = json.dumps(receipt).encode()
    with pytest.raises(ValueError):
        bootstrap.fetch_ready_manifest(client, **options)
    assert not options["manifest_path"].exists()
    assert all(body.closed for body in client.bodies)


@pytest.mark.parametrize("damage", ["pilot", "unverified", "missing_scene", "wrong_split"])
def test_receipt_cannot_claim_full_coverage_for_incomplete_manifest(handoff, damage):
    client, receipt, manifest, options = handoff
    if damage == "pilot":
        manifest["provenance"]["coverage"] = "pilot"
    elif damage == "unverified":
        manifest["provenance"]["remote_verified"] = False
    elif damage == "missing_scene":
        manifest["scenes"].pop()
    else:
        manifest["split"] = "test"
    payload = json.dumps(manifest).encode()
    receipt["manifest"].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    client.objects[receipt["manifest"]["key"]] = payload
    client.objects[options["ready_key"]] = json.dumps(receipt).encode()
    with pytest.raises(ValueError, match="full, remotely verified"):
        bootstrap.fetch_ready_manifest(client, **options)
    assert not options["manifest_path"].exists()


def test_production_invocation_records_waiver_and_preserves_runner_failure(handoff, monkeypatch):
    client, receipt, _, options = handoff
    monkeypatch.setattr(bootstrap, "metadata_client", lambda endpoint: client)
    invocations = []

    def run(command, *, check):
        assert not check
        invocations.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert (
        bootstrap.main(
            [
                "--nat-bucket",
                options["bucket"],
                "--nat-endpoint",
                options["endpoint"],
                "--nat-ready-key",
                options["ready_key"],
                "--nat-manifest-path",
                str(options["manifest_path"]),
                "--nat-inventory",
                str(options["inventory_path"]),
                "--",
                "--segment-hours",
                "2",
                "--budget-usd",
                "32",
            ]
        )
        == 7
    )
    command = invocations[0]
    assert (
        command[command.index("--naturalistic-manifest-sha256") + 1]
        == receipt["manifest"]["sha256"]
    )
    assert (
        command[command.index("--streaming-acceptance-waiver-reason") + 1]
        == bootstrap.WAIVER_REASON
    )
    assert "--nat" in command and "--skip-streaming-acceptance" in command
    assert "--streaming-smoke-report" not in command
    assert command[-4:] == ["--segment-hours", "2", "--budget-usd", "32"]


def test_bootstrap_inputs_cannot_be_replaced_and_existing_manifest_is_preserved(handoff):
    client, _, _, options = handoff
    path = options["manifest_path"]
    path.parent.mkdir()
    path.write_bytes(b"previous frozen input")
    with pytest.raises(FileExistsError):
        bootstrap.fetch_ready_manifest(client, **options)
    assert path.read_bytes() == b"previous frozen input"
    with pytest.raises(ValueError, match="override"):
        bootstrap.segment_command(
            ["--naturalistic-manifest=another.json"],
            manifest_path=path,
            manifest_sha256="a" * 64,
            endpoint=options["endpoint"],
        )


def test_metadata_timeout_is_bounded_before_network(handoff):
    client, _, _, options = handoff
    invalid = deepcopy(options)
    invalid["timeout_seconds"] = float("inf")
    with pytest.raises(ValueError, match="timeout"):
        bootstrap.fetch_ready_manifest(client, **invalid)
    assert client.reads == []


def test_expired_queue_start_never_creates_client_or_launches_training(handoff, monkeypatch):
    _, _, _, options = handoff

    def forbidden(*args, **kwargs):
        pytest.fail("Expired production start performed a network or training operation")

    monkeypatch.setattr(bootstrap, "metadata_client", forbidden)
    monkeypatch.setattr(bootstrap.subprocess, "run", forbidden)
    with pytest.raises(TimeoutError, match="start deadline"):
        bootstrap.main(
            [
                "--nat-bucket",
                options["bucket"],
                "--nat-ready-key",
                options["ready_key"],
                "--nat-manifest-path",
                str(options["manifest_path"]),
                "--latest-start-utc",
                "2000-01-01T00:00:00Z",
                "--",
                "--segment-hours",
                "2",
            ]
        )


def test_start_deadline_uses_utc_and_rejects_exact_boundary():
    latest = bootstrap.utc_timestamp("2026-10-26T00:00:00Z")
    bootstrap.ensure_start_deadline(
        latest, now=datetime(2026, 10, 25, 23, 59, 59, tzinfo=timezone.utc)
    )
    with pytest.raises(TimeoutError, match="deadline"):
        bootstrap.ensure_start_deadline(latest, now=latest)
    with pytest.raises(argparse.ArgumentTypeError, match="UTC"):
        bootstrap.utc_timestamp("2026-10-26T00:00:00")


def test_deadline_is_rechecked_after_readiness_download(handoff, monkeypatch):
    client, _, _, options = handoff
    monkeypatch.setattr(bootstrap, "metadata_client", lambda endpoint: client)
    checks = []

    def deadline(_latest):
        checks.append(True)
        if len(checks) == 2:
            raise TimeoutError("start deadline passed while downloading metadata")

    monkeypatch.setattr(bootstrap, "ensure_start_deadline", deadline)
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched")
    )
    with pytest.raises(TimeoutError, match="downloading"):
        bootstrap.main(
            [
                "--nat-bucket",
                options["bucket"],
                "--nat-endpoint",
                options["endpoint"],
                "--nat-ready-key",
                options["ready_key"],
                "--nat-manifest-path",
                str(options["manifest_path"]),
                "--nat-inventory",
                str(options["inventory_path"]),
                "--",
                "--segment-hours",
                "2",
            ]
        )
    assert len(checks) == 2

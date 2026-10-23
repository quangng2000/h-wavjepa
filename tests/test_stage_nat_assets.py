"""CPU-only fixtures: no network, paid compute, or publisher downloads."""

import contextlib
import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
import threading
from types import SimpleNamespace

import pytest

import src.stage_nat_assets as staging
from src.prepare_nat_assets import PreparationLimit, _atomic_json, assemble_nat_manifest


class S3Error(Exception):
    def __init__(self, status):
        self.response = {
            "Error": {"Code": str(status)},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.corrupt_get = False

    def head_object(self, *, Bucket, Key):
        self.calls.append(("head", Key))
        if Key not in self.objects:
            raise S3Error(404)
        value = self.objects[Key]
        return {"ContentLength": len(value), "ETag": hashlib.sha256(value).hexdigest()}

    def get_object(self, *, Bucket, Key):
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise S3Error(404)
        original = self.objects[Key]
        value = b"X" * len(original) if self.corrupt_get else original
        return {
            "Body": io.BytesIO(value),
            "ContentLength": len(value),
            "ETag": hashlib.sha256(original).hexdigest(),
        }

    def put_object(self, *, Bucket, Key, Body, ContentLength, IfNoneMatch=None, **kwargs):
        self.calls.append(("put", Key))
        if IfNoneMatch == "*" and Key in self.objects:
            raise S3Error(412)
        payload = Body.read() if hasattr(Body, "read") else Body
        assert len(payload) == ContentLength
        self.objects[Key] = payload
        return {}


def test_pinned_inventory_has_seventy_brirs_and_publisher_md5():
    sources = staging.read_inventory(staging.INVENTORY)
    assert len(sources) == 71
    assert sources[-1]["md5"] == staging.WHAM_MD5
    assert sources[-1]["bytes"] == staging.WHAM_BYTES
    assert staging.BRIR_REVISION in sources[0]["url"]


@pytest.mark.parametrize("mutation", ["revision", "path", "size", "md5", "url"])
def test_inventory_rejects_source_substitution(tmp_path, mutation):
    value = json.loads(staging.INVENTORY.read_text())
    if mutation == "revision":
        value["brir"]["revision"] = "main"
    elif mutation == "path":
        value["brir"]["shards"][0]["path"] = "../evil.tar"
    elif mutation == "size":
        value["brir"]["shards"][0]["bytes"] = 0
    elif mutation == "md5":
        value["noise"]["published_md5"] = "0" * 32
    else:
        value["noise"]["download_url"] = "https://untrusted.example/file"
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        staging.read_inventory(path)


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload, declared=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if declared is None else declared)}


@pytest.mark.parametrize("kind", ["sha256", "md5"])
def test_download_verifies_publisher_digest_and_reuses_promoted_archive(tmp_path, kind):
    data = b"source fixture"
    source = {
        "url": "https://publisher.example/data",
        "bytes": len(data),
        kind: hashlib.new(kind, data).hexdigest(),
    }
    path = tmp_path / "source.zip"
    proof = staging.download_source(
        source, path, lambda: None, opener=lambda *a, **k: Response(data)
    )
    assert proof["publisher_digest_verified"]
    assert proof["sha256"] == hashlib.sha256(data).hexdigest()
    assert path.read_bytes() == data
    assert not path.with_suffix(".zip.part").exists()
    assert (
        staging.download_source(
            source, path, lambda: None, opener=lambda *a, **k: pytest.fail("redownload")
        )
        == proof
    )


@pytest.mark.parametrize("fault", ["digest", "length", "oversize", "truncated"])
def test_failed_download_never_promotes_archive(tmp_path, fault):
    data = b"fixture"
    source = {
        "url": "https://publisher.example/data",
        "bytes": len(data),
        "md5": hashlib.md5(data).hexdigest(),
    }
    if fault == "digest":
        source["md5"] = "0" * 32
    response = Response(
        data + b"x" if fault == "oversize" else data[:-1] if fault == "truncated" else data,
        declared=100 if fault == "length" else len(data),
    )
    path = tmp_path / "source.zip"
    with pytest.raises(ValueError):
        staging.download_source(source, path, lambda: None, opener=lambda *a, **k: response)
    assert not path.exists()


def test_new_payload_uses_two_requests_and_full_get_proof(tmp_path):
    client = FakeS3()
    path = tmp_path / "payload.npz"
    path.write_bytes(b"converted fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = staging.NatStore(client, "bucket", lambda: None)
    receipt = store.put_payload("objects/item", path, digest)
    assert [call[0] for call in client.calls] == ["put", "get"]
    assert receipt["sha256"] == digest
    store.put_payload("objects/item", path, digest)
    assert [call[0] for call in client.calls[-2:]] == ["put", "get"]
    client.corrupt_get = True
    with pytest.raises(ValueError, match="SHA-256"):
        store.put_payload("objects/item", path, digest)


def test_limiter_retries_throttle_and_rewinds_put_body(monkeypatch):
    class Throttled:
        calls = 0

        def put_object(self, **kwargs):
            self.calls += 1
            assert kwargs["Body"].read() == b"full payload"
            if self.calls == 1:
                raise S3Error(429)
            return {"ok": True}

    client = Throttled()
    limiter = staging.LimitedClient(client, lambda: None)
    delays = []
    monkeypatch.setattr(limiter, "_wait", delays.append)
    assert limiter.put_object(Body=io.BytesIO(b"full payload")) == {"ok": True}
    assert client.calls == 2 and 1 in delays
    with pytest.raises(ValueError):
        staging.LimitedClient(client, lambda: None, 10)


@pytest.mark.parametrize("failure", ["ReadTimeoutError", "ConnectionClosedError"])
def test_transport_failure_with_no_response_retries_and_rewinds_body(monkeypatch, failure):
    errors = pytest.importorskip("botocore.exceptions")
    error = getattr(errors, failure)(endpoint_url="https://signed.example/?secret=private")
    assert error.response is None

    class Interrupted:
        calls = 0

        def put_object(self, **kwargs):
            self.calls += 1
            assert kwargs["Body"].read() == b"entire object"
            if self.calls == 1:
                raise error
            return {"ok": True}

    client = Interrupted()
    limiter = staging.LimitedClient(client, lambda: None)
    delays = []
    monkeypatch.setattr(limiter, "_wait", delays.append)
    assert limiter.put_object(Body=io.BytesIO(b"entire object")) == {"ok": True}
    assert client.calls == 2 and 1 in delays


def test_exhausted_transport_retries_preserve_original_failure(monkeypatch):
    errors = pytest.importorskip("botocore.exceptions")
    error = errors.ReadTimeoutError(endpoint_url="https://signed.example/?secret=private")

    class Interrupted:
        calls = 0

        def head_object(self, **kwargs):
            self.calls += 1
            raise error

    client = Interrupted()
    limiter = staging.LimitedClient(client, lambda: None)
    monkeypatch.setattr(limiter, "_wait", lambda seconds: None)
    store = staging.NatStore(limiter, "bucket", lambda: None)
    with pytest.raises(errors.ReadTimeoutError) as caught:
        store.head("component")
    assert caught.value is error and client.calls == 6


@pytest.mark.parametrize("operation", ["payload", "control"])
def test_immutable_put_preserves_transport_failure_without_response(tmp_path, operation):
    errors = pytest.importorskip("botocore.exceptions")
    error = errors.ReadTimeoutError(endpoint_url="https://signed.example/?secret=private")

    class Interrupted:
        def put_object(self, **kwargs):
            raise error

    store = staging.NatStore(Interrupted(), "bucket", lambda: None)
    path = tmp_path / "payload"
    path.write_bytes(b"converted fixture")
    with pytest.raises(errors.ReadTimeoutError) as caught:
        if operation == "payload":
            store.put_payload("key", path, staging.sha256_file(path))
        else:
            store.put_json("key", {"value": True}, immutable=True)
    assert caught.value is error


def test_full_get_restarts_proof_after_interrupted_stream(monkeypatch):
    errors = pytest.importorskip("botocore.exceptions")
    payload = b"full object"

    class InterruptedBody(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise errors.ResponseStreamingError(error="private signed URL")
            return super().read(3)

    interrupted = InterruptedBody(payload)

    class Interrupted:
        calls = 0

        def get_object(self, **kwargs):
            self.calls += 1
            return {
                "Body": interrupted if self.calls == 1 else io.BytesIO(payload),
                "ContentLength": len(payload),
                "ETag": "verified",
            }

    client = Interrupted()
    delays = []
    monkeypatch.setattr(staging, "_wait_checked", lambda check, delay: delays.append(delay))
    store = staging.NatStore(client, "bucket", lambda: None)
    digest = hashlib.sha256(payload).hexdigest()
    assert store.verify("key", digest, len(payload))["sha256"] == digest
    assert interrupted.closed and client.calls == 2 and delays == [1]


def test_failure_diagnostics_preserve_frames_without_sensitive_content(factory):
    secret = "https://signed.example/?secret=never-serialize-this"
    try:
        try:
            raise RuntimeError(secret)
        except RuntimeError as cause:
            raise AttributeError(secret) from cause
    except AttributeError as error:
        diagnostic = factory().status("failed", failure=error)
    encoded = json.dumps(diagnostic)
    assert "signed.example" not in encoded and "never-serialize-this" not in encoded
    assert diagnostic["failure_type"] == "AttributeError"
    assert [item["type"] for item in diagnostic["failure_trace"]] == [
        "AttributeError",
        "RuntimeError",
    ]
    assert diagnostic["failure_trace"][0]["frames"][-1] == {
        "file": "test_stage_nat_assets.py",
        "function": "test_failure_diagnostics_preserve_frames_without_sensitive_content",
        "line": diagnostic["failure_trace"][0]["frames"][-1]["line"],
    }


def test_immutable_control_publication_never_overwrites_competing_content(tmp_path):
    client = FakeS3()
    store = staging.NatStore(client, "bucket", lambda: None)
    store.put_json("ready.json", {"ready": True}, immutable=True)
    original = client.objects["ready.json"]
    store.put_json("ready.json", {"ready": True}, immutable=True)
    with pytest.raises(ValueError):
        store.put_json("ready.json", {"ready": False}, immutable=True)
    assert client.objects["ready.json"] == original
    path = tmp_path / "manifest.json"
    path.write_bytes(b"manifest one")
    store.put_file("manifest.json", path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(b"manifest two")
    with pytest.raises(ValueError, match="SHA-256"):
        store.put_file("manifest.json", path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert client.objects["manifest.json"] == b"manifest one"


def component_fixture(directory, *, count=1000):
    """Many logical scenes share two tiny content-addressed objects."""
    (directory / "objects").mkdir(parents=True)
    component = {
        "schema_version": 1,
        "status": "complete",
        "kind": "nat_brir_shard",
        "sample_rate": 16000,
        "split": "train",
        "converted_count": count,
        "source": {
            "name": "source.tar",
            "bytes": 7,
            "sha256": hashlib.sha256(b"archive").hexdigest(),
            "verified": True,
        },
        "provenance": {
            "repository": staging.BRIR_REPOSITORY,
            "revision": staging.BRIR_REVISION,
            "path": "train/webdataset_output/shard-000000.tar",
            "source_sample_rate": 32000,
        },
        "scenes": [],
        "noise_files": [],
        "assets": {},
        "output_bytes": 0,
    }
    for variant, payload in enumerate((b"localized packed", b"diffuse packed")):
        digest = hashlib.sha256(payload).hexdigest()
        key = f"objects/{digest}.npz"
        (directory / key).write_bytes(payload)
        component["output_bytes"] += len(payload)
        for scene_index in range(variant, count, 2):
            identity = f"scene-{scene_index}"
            names = [f"brirs/{identity}/{index}.npz" for index in range(2 if variant == 0 else 4)]
            component["scenes"].append(
                {
                    "id": identity,
                    "noise_type": "localized" if variant == 0 else "diffuse",
                    "source_rir": names[0],
                    "noise_rirs": names[1:],
                }
            )
            for index, name in enumerate(names):
                component["assets"][name] = {
                    "key": key,
                    "bytes": len(payload),
                    "sha256": digest,
                    "index": index,
                    "array_key": "arr_0",
                }
    _atomic_json(directory / "manifest.json", component)
    return component


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setattr(staging, "LimitedClient", lambda client, *args: client)
    monkeypatch.setattr(
        staging.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 1024**3)
    )
    created = []

    def make(client=None, output=None, **kwargs):
        result = staging.NatStager(
            output or tmp_path / f"job-{len(created)}",
            client or FakeS3(),
            endpoint="https://storage.example.com",
            bucket="nat-assets",
            prefix="nat-assets/20261023-v1",
            **kwargs,
        )
        created.append(result)
        return result

    yield make
    for item in created:
        with contextlib.suppress(Exception):
            item.close()


def install_source_fixture(stager, monkeypatch):
    source = {
        "id": "shard-000000",
        "kind": "brir",
        "bytes": 7,
        "sha256": hashlib.sha256(b"archive").hexdigest(),
        "path": "train/webdataset_output/shard-000000.tar",
        "url": "https://fixture.example/archive",
    }
    real_download = staging.download_source
    monkeypatch.setattr(
        staging,
        "download_source",
        lambda src, path, check: real_download(
            src, path, check, opener=lambda *a, **k: Response(b"archive")
        ),
    )
    monkeypatch.setattr(
        staging,
        "prepare_brir_shard",
        lambda archive, digest, directory, **kwargs: component_fixture(directory),
    )
    stager.sources = [source]
    return source


def test_stager_enrolls_readback_then_cleans_and_resumes_ephemeral_job(factory, monkeypatch):
    client = FakeS3()
    first = factory(client)
    source = install_source_fixture(first, monkeypatch)
    result = first.run()
    assert result["status"] == "partial" and result["scene_count"] == 1000
    assert not (first.scratch / source["id"]).exists()
    assert f"{first.prefix}/receipts/{source['id']}.json" in client.objects
    assert f"{first.prefix}/manifest.json" not in client.objects
    second = factory(client)
    second.sources = [source]
    monkeypatch.setattr(
        staging,
        "download_source",
        lambda *a, **k: pytest.fail("verified remote component redownloaded"),
    )
    calls = len(client.calls)
    assert second.run()["scene_count"] == 1000
    assert all(method != "put" for method, _ in client.calls[calls:])
    assert second.database.execute("SELECT count(*) FROM objects").fetchone()[0] == 2


def test_failed_get_keeps_scratch_and_never_enrolls_component(factory, monkeypatch):
    client = FakeS3()
    stager = factory(client)
    source = install_source_fixture(stager, monkeypatch)
    client.corrupt_get = True
    with pytest.raises(ValueError, match="SHA-256"):
        stager.run()
    assert (stager.scratch / source["id"] / "source.tar").exists()
    assert stager.database.execute("SELECT count(*) FROM components").fetchone()[0] == 0
    assert not any(key.endswith("ready.json") for key in client.objects)
    assert json.loads((stager.output / "status.json").read_text())["status"] == "failed"


def test_remote_resume_rejects_missing_previously_verified_object(factory, monkeypatch):
    client = FakeS3()
    stager = factory(client)
    source = install_source_fixture(stager, monkeypatch)
    stager.run()
    del client.objects[next(key for key in client.objects if "/objects/" in key)]
    resumed = factory(client)
    resumed.sources = [source]
    with pytest.raises(ValueError, match="missing"):
        resumed.run()


def test_bounds_leave_partial_without_publishing_manifest(factory, monkeypatch):
    stager = factory(max_upload_bytes=1)
    source = install_source_fixture(stager, monkeypatch)
    with pytest.raises(PreparationLimit, match="byte ceiling"):
        stager.run()
    assert (stager.scratch / source["id"]).exists()
    assert stager.status("partial")["completed_components"] == 0
    stager.cancelled.clear()
    stager.deadline = 0
    with pytest.raises(PreparationLimit, match="runtime"):
        stager.check()


def test_diagnostics_capture_compact_ledger_without_scratch_payloads(
    factory, monkeypatch, tmp_path
):
    stager = factory(max_upload_bytes=1)
    install_source_fixture(stager, monkeypatch)
    with pytest.raises(PreparationLimit):
        stager.run()
    destination = tmp_path / "job-artifact"
    stager.preserve_diagnostics(destination)
    assert {path.name for path in destination.iterdir()} == {
        "configuration.json",
        "status.json",
        "ledger.sqlite3",
    }
    with sqlite3.connect(destination / "ledger.sqlite3") as database:
        assert database.execute("SELECT count(*) FROM components").fetchone()[0] == 0
    assert json.loads((destination / "status.json").read_text())["status"] == "partial"


def test_final_assembly_remote_route_checks_receipts_without_local_payloads(factory, tmp_path):
    stager = factory()
    directory = tmp_path / "brir"
    component = component_fixture(directory, count=2)
    stager._upload(component, directory)
    shutil.rmtree(directory / "objects")
    noise_root = tmp_path / "noise"
    (noise_root / "objects").mkdir(parents=True)
    payload = b"noise fixture"
    digest = hashlib.sha256(payload).hexdigest()
    key = f"objects/{digest}.wav"
    (noise_root / key).write_bytes(payload)
    noise = {
        "schema_version": 1,
        "status": "complete",
        "kind": "nat_wham_noise",
        "sample_rate": 16000,
        "split": "train",
        "converted_count": 1,
        "source": {"bytes": 1, "sha256": "0" * 64, "verified": True},
        "provenance": {
            "publisher": "http://wham.whisper.ai/",
            "split": "wham_noise/tr/ only",
            "channel_policy": "first channel, index 0",
        },
        "scenes": [],
        "noise_files": ["noise/one.wav"],
        "output_bytes": len(payload),
        "assets": {
            "noise/one.wav": {
                "key": key,
                "bytes": len(payload),
                "sha256": digest,
                "source_member": "wham_noise/tr/one.wav",
            }
        },
    }
    _atomic_json(noise_root / "manifest.json", noise)
    stager._upload(noise, noise_root)
    shutil.rmtree(noise_root / "objects")
    result = assemble_nat_manifest(
        [directory / "manifest.json"],
        noise_root / "manifest.json",
        tmp_path / "assembled.json",
        endpoint=stager.endpoint,
        bucket=stager.bucket,
        prefix=stager.prefix,
        allow_pilot=True,
        _remote_verifier=stager.verify_enrolled,
    )
    assert result["status"] == "remote_objects_verified_manifest_not_published"
    content = json.loads((tmp_path / "assembled.json").read_text())
    assert content["provenance"]["remote_verified"]
    assert content["provenance"]["coverage"] == "pilot"
    stager.database.execute("DELETE FROM objects")
    with pytest.raises(ValueError, match="receipt"):
        assemble_nat_manifest(
            [directory / "manifest.json"],
            noise_root / "manifest.json",
            tmp_path / "bad.json",
            endpoint=stager.endpoint,
            bucket=stager.bucket,
            prefix=stager.prefix,
            allow_pilot=True,
            _remote_verifier=stager.verify_enrolled,
        )


def two_sources(stager, monkeypatch):
    first = install_source_fixture(stager, monkeypatch)
    second = {**first, "id": "shard-000001", "path": "train/webdataset_output/shard-000001.tar"}
    stager.sources = [first, second]
    return first, second


def test_pipeline_prepares_next_source_during_upload_and_keeps_ledger_on_main(factory, monkeypatch):
    stager = factory()
    first, second = two_sources(stager, monkeypatch)
    prepare, publish = stager._prepare_source, stager._publish_source
    uploading, prepared_next = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    active_counts = []

    def producer(source):
        assert threading.get_ident() != main_thread
        active_counts.append(len(stager.reservations))
        if source["id"] == second["id"]:
            assert uploading.wait(5), "next preparation did not overlap current upload"
        result = prepare(source)
        if source["id"] == second["id"]:
            prepared_next.set()
        return result

    def consumer(prepared):
        assert threading.get_ident() == main_thread
        if prepared["source"]["id"] == first["id"]:
            uploading.set()
            assert prepared_next.wait(5), "consumer blocked the producer"
        return publish(prepared)

    monkeypatch.setattr(stager, "_prepare_source", producer)
    monkeypatch.setattr(stager, "_publish_source", consumer)
    assert stager.run()["completed_components"] == 2
    assert active_counts == [1, 2]
    assert stager.reservations == {}


@pytest.mark.parametrize(
    "options", [{"prefetch_sources": 0}, {"max_scratch_bytes": 7 * 1024**3 // 2}]
)
def test_pipeline_can_run_sequentially_or_fall_back_for_small_disk(factory, monkeypatch, options):
    stager = factory(**options)
    first, second = two_sources(stager, monkeypatch)
    prepare, publish = stager._prepare_source, stager._publish_source
    first_published = threading.Event()

    def producer(source):
        assert len(stager.reservations) == 1
        if source["id"] == second["id"]:
            assert first_published.is_set()
        return prepare(source)

    def consumer(prepared):
        result = publish(prepared)
        if prepared["source"]["id"] == first["id"]:
            first_published.set()
        return result

    monkeypatch.setattr(stager, "_prepare_source", producer)
    monkeypatch.setattr(stager, "_publish_source", consumer)
    assert stager.run()["completed_components"] == 2


def test_consumer_failure_cancels_background_preparation_and_preserves_scratch(
    factory, monkeypatch
):
    stager = factory()
    first, second = two_sources(stager, monkeypatch)
    original = stager._prepare_source
    started, stopped = threading.Event(), threading.Event()

    def producer(source):
        if source["id"] == first["id"]:
            return original(source)
        started.set()
        try:
            while not stager.cancelled.wait(0.05):
                stager.check()
            stager.check()
        finally:
            stopped.set()

    def failed_upload(prepared):
        assert started.wait(5)
        raise ValueError("fixture upload failure")

    monkeypatch.setattr(stager, "_prepare_source", producer)
    monkeypatch.setattr(stager, "_publish_source", failed_upload)
    with pytest.raises(ValueError, match="fixture upload"):
        stager.run()
    assert stopped.is_set() and not stager.reservations
    assert (stager.scratch / first["id"] / "source.tar").exists()
    assert stager.database.execute("SELECT count(*) FROM components").fetchone()[0] == 0


def test_upload_refills_worker_without_waiting_for_slowest_batch_member(
    factory, monkeypatch, tmp_path
):
    stager = factory(workers=2)
    assets = {}
    keys = []
    for index in range(3):
        payload = f"payload-{index}".encode()
        digest = hashlib.sha256(payload).hexdigest()
        local_key = f"objects/{digest}.npz"
        path = tmp_path / local_key
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        assets[str(index)] = {"key": local_key, "bytes": len(payload), "sha256": digest}
        keys.append(f"{stager.prefix}/{local_key}")
    third_started = threading.Event()
    put = stager.store.put_payload

    def slow_first(key, path, digest):
        if key == keys[0]:
            assert third_started.wait(5), "upload queue still has a batch barrier"
        if key == keys[2]:
            third_started.set()
        return put(key, path, digest)

    monkeypatch.setattr(stager.store, "put_payload", slow_first)
    stager._upload({"assets": assets}, tmp_path)
    assert stager.database.execute("SELECT count(*) FROM objects").fetchone()[0] == 3


def test_huggingface_supervisor_promotes_one_archive_and_strips_private_environment(
    factory, monkeypatch
):
    stager = factory(download_backend="huggingface")
    source = install_source_fixture(stager, monkeypatch)
    stager._reserve_source(source)
    work = stager.scratch / source["id"]
    work.mkdir()
    archive = work / "source.tar"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private fixture credential")
    monkeypatch.setenv("GMN_API_TOKEN", "private fixture token")
    monkeypatch.setenv("HF_TOKEN", "private fixture token")
    calls = []

    def launch(arguments, **kwargs):
        assert kwargs["start_new_session"]
        assert not any(key.startswith(("AWS_", "GMN_", "HF_TOKEN")) for key in kwargs["env"])
        assert kwargs["stdout"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.DEVNULL
        destination = work / "huggingface" / source["path"]
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"archive")
        calls.append(arguments)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(staging.subprocess, "Popen", launch)
    stager._download_brir_huggingface(source, archive)
    proof = staging.download_source(source, archive, stager.check)
    assert proof["sha256"] == source["sha256"]
    assert len(calls) == 1 and not (work / "huggingface" / source["path"]).exists()


def test_huggingface_supervisor_kills_process_group_on_cancellation(factory, monkeypatch):
    stager = factory(download_backend="huggingface")
    source = install_source_fixture(stager, monkeypatch)
    stager._reserve_source(source)
    work = stager.scratch / source["id"]
    work.mkdir()
    killed = []
    process = SimpleNamespace(pid=1234, poll=lambda: None, wait=lambda timeout=None: 0)

    def launch(*args, **kwargs):
        stager.cancelled.set()
        return process

    monkeypatch.setattr(staging.subprocess, "Popen", launch)
    monkeypatch.setattr(staging.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(PreparationLimit, match="cancelled"):
        stager._download_brir_huggingface(source, work / "source.tar")
    assert killed == [(1234, staging.signal.SIGTERM)]

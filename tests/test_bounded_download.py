"""Real SDK range downloads isolated from hung transports and scoped credentials."""

import contextlib
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
import psutil
import pytest
from botocore.config import Config

from scripts import checkpoint_transfer as transfer


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "audio-test-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "audio-test-secret")
    monkeypatch.setenv("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID", "checkpoint-test-access")
    monkeypatch.setenv("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY", "checkpoint-test-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    state = {"objects": {}, "ranges": [], "keys": [], "active": 0, "peak": 0, "corrupt": False}
    lock = threading.Lock()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            payload = state["objects"].get(self.path)
            self.send_response(200 if payload is not None else 404)
            self.send_header("Content-Length", str(len(payload) if payload is not None else 0))
            self.send_header("ETag", '"test-etag"')
            self.end_headers()

        def do_GET(self):
            payload = state["objects"].get(self.path)
            if payload is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["keys"].append(self.headers.get("Authorization", ""))
            try:
                if state.get("block"):
                    release.wait(15)
                time.sleep(0.04)
                raw_range = self.headers.get("Range")
                if state["corrupt"]:
                    payload = b"x" + payload[1:]
                if raw_range:
                    start, stop = raw_range.removeprefix("bytes=").split("-")
                    start, stop = int(start), int(stop) if stop else len(payload) - 1
                    with lock:
                        state["ranges"].append((start, stop))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{stop}/{len(payload)}")
                    payload = payload[start : stop + 1]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("ETag", '"test-etag"')
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(payload)
            finally:
                with lock:
                    state["active"] -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        region_name="auto",
        aws_access_key_id="unused-parent-access",
        aws_secret_access_key="unused-parent-secret",
        config=Config(s3={"addressing_style": "path"}),
    )
    try:
        yield client, state
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def download(client, destination, payload, *, scope="audio", timeout=15):
    transfer.download_verified_object(
        client,
        "private",
        "object",
        destination,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        timeout_seconds=timeout,
        credential_scope=scope,
    )


def test_real_sdk_parallel_ranges_validate_and_atomically_publish(gateway, tmp_path, capfd):
    client, state = gateway
    payload = b"verified-range-" * (3 * transfer.DOWNLOAD_PART_BYTES // 15 + 1)
    state["objects"]["/private/object"] = payload
    destination = tmp_path / "verified.tar"
    download(client, destination, payload)
    assert destination.read_bytes() == payload
    assert len(state["ranges"]) >= 3
    assert state["peak"] >= 2
    assert all("audio-test-access" in value for value in state["keys"])
    assert not list(tmp_path.glob(".verified-download-*"))
    output = capfd.readouterr().out
    assert '"phase": "completed"' in output
    assert "audio-test-secret" not in output


def test_checkpoint_client_scope_does_not_fall_back_to_audio(gateway, tmp_path):
    client, state = gateway
    payload = b"full-training-state"
    state["objects"]["/private/object"] = payload
    download(client, tmp_path / "resume.ckpt", payload, scope="checkpoint")
    assert all("checkpoint-test-access" in value for value in state["keys"])


def test_missing_checkpoint_credentials_never_use_audio_credentials(gateway, tmp_path, monkeypatch):
    client, state = gateway
    monkeypatch.delenv("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY")
    state["objects"]["/private/object"] = b"checkpoint"
    with pytest.raises(RuntimeError, match="Managed object download failed"):
        download(client, tmp_path / "resume.ckpt", b"checkpoint", scope="checkpoint")
    assert not state["keys"]


def test_corruption_never_becomes_visible_and_removes_sdk_scratch(gateway, tmp_path):
    client, state = gateway
    payload = b"verified-object"
    state["objects"]["/private/object"] = payload
    state["corrupt"] = True
    destination = tmp_path / "verified.tar"
    with pytest.raises(RuntimeError, match="Managed object download failed"):
        download(client, destination, payload)
    assert not destination.exists()
    assert not list(tmp_path.glob(".verified-download-*"))


def test_indexed_size_mismatch_refused_before_body_transfer(gateway, tmp_path):
    client, state = gateway
    state["objects"]["/private/object"] = b"unexpected-long-object"
    with pytest.raises(RuntimeError, match="Managed object download failed"):
        download(client, tmp_path / "verified.tar", b"short")
    assert not state["keys"]
    assert not list(tmp_path.glob(".verified-download-*"))


def test_hung_sdk_threads_are_killed_reaped_and_scratch_removed(gateway, tmp_path, monkeypatch):
    client, _ = gateway
    original = subprocess.Popen
    children = []

    def stuck_child(argv, **kwargs):
        request = json.loads(argv[-1])
        code = (
            "import pathlib,signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({request['path']!r}+'.sdk-temp').write_bytes(b'partial'); "
            "time.sleep(30)"
        )
        child = original([sys.executable, "-c", code], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(transfer.subprocess, "Popen", stuck_child)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="total deadline"):
        download(client, tmp_path / "verified.tar", b"object", timeout=2)
    assert time.monotonic() - started < 3
    assert children[0].returncode == -signal.SIGKILL
    assert not (tmp_path / "verified.tar").exists()
    assert not list(tmp_path.glob(".verified-download-*"))
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


def test_download_child_does_not_outlive_abruptly_terminated_worker(gateway, tmp_path):
    client, state = gateway
    state["objects"]["/private/object"] = b"object"
    state["block"] = True
    code = (
        "import boto3; from scripts.checkpoint_transfer import download_verified_object; "
        f"c=boto3.client('s3',endpoint_url={client.meta.endpoint_url!r},region_name='auto'); "
        f"download_verified_object(c,'private','object',{str(tmp_path / 'data.tar')!r},"
        f"size=6,sha256={hashlib.sha256(b'object').hexdigest()!r},"
        "timeout_seconds=30,credential_scope='audio')"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and child_pid is None:
            if select.select([parent.stdout], [], [], 0.1)[0]:
                line = parent.stdout.readline()
                if line:
                    event = json.loads(line)
                    if event.get("phase") == "started":
                        child_pid = event["pid"]
        assert child_pid is not None
        parent.terminate()  # SIGTERM deliberately bypasses the parent's finally block.
        parent.wait(timeout=2)
        deadline = time.monotonic() + 3
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            if psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert (
            not psutil.pid_exists(child_pid)
            or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        )
        assert not (tmp_path / "data.tar").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=2)
        parent.stdout.close()
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_real_shard_cache_uses_managed_process_transport(gateway, tmp_path):
    from src.data.sharded_audio_datamodule import _ShardCache

    client, state = gateway
    payload = b"small-shard-transport-fixture"
    state["objects"]["/private/object"] = payload
    shard = {"key": "object", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    cache = _ShardCache(tmp_path, client, "private", 10000, 0, download_timeout_seconds=15)
    result = cache.get(shard)
    assert result.read_bytes() == payload
    before = len(state["keys"])
    assert cache.get(shard) == result
    assert len(state["keys"]) == before


def test_repository_resume_uses_pinned_receipt_and_checkpoint_scope(gateway, tmp_path):
    from scripts.segment_checkpoint import CheckpointRepository

    client, state = gateway
    repo = CheckpointRepository(
        client, "private", "training/verified-run", "verified-run", "a" * 64, check=lambda: None
    )
    payload = b"full-optimizer-state-fixture"
    sha = hashlib.sha256(payload).hexdigest()
    key = f"{repo.prefix}/checkpoints/step_0000250_{sha}.ckpt"
    receipt = {
        "owner": repo.owner,
        "checkpoint": {"step": 250, "key": key, "sha256": sha, "bytes": len(payload)},
        "previous": None,
        "resolved_data_seed": 42,
        "stream_positions_checkpointed": False,
    }
    state["objects"].update(
        {
            f"/private/{repo.owner_key}": json.dumps(repo.owner).encode(),
            f"/private/{repo.latest_key}": json.dumps(receipt).encode(),
            f"/private/{key}": payload,
        }
    )
    result = repo.download_latest(tmp_path / "resume.ckpt", timeout_seconds=15)
    assert result == receipt
    assert (tmp_path / "resume.ckpt").read_bytes() == payload
    assert "checkpoint-test-access" in state["keys"][-1]

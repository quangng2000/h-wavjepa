"""Exercise managed multipart transfers through the real SDK and local HTTP."""

import contextlib
import hashlib
import json
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from scripts.checkpoint_transfer import (
    ManagedCheckpointStore,
    _UploadProgress,
    safe_exception_details,
)


@pytest.fixture
def gateway():
    state = {
        "objects": {},
        "parts": {},
        "part_numbers": [],
        "created": 0,
        "completed": 0,
        "aborted": 0,
        "readbacks": 0,
        "fail_parts": False,
        "fail_complete": False,
        "corrupt_readback": False,
    }
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, code, body=b"", *, size=None, etag=None):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body) if size is None else size))
            if etag is not None:
                self.send_header("ETag", etag)
            self.end_headers()
            if self.command != "HEAD":
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

        def parse(self):
            parsed = urlsplit(self.path)
            return parsed.path, parse_qs(parsed.query, keep_blank_values=True)

        def do_HEAD(self):
            key, _ = self.parse()
            payload = state["objects"].get(key)
            if payload is None:
                self.respond(404)
            else:
                self.respond(200, size=len(payload), etag='"verified-object"')

        def do_GET(self):
            key, _ = self.parse()
            payload = state["objects"].get(key)
            if payload is None:
                self.respond(404)
            else:
                state["readbacks"] += 1
                if state["corrupt_readback"]:
                    payload = b"x" + payload[1:]
                self.respond(200, payload)

        def do_POST(self):
            key, query = self.parse()
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if "uploads" in query:
                state["created"] += 1
                self.respond(
                    200,
                    b"<InitiateMultipartUploadResult><UploadId>test-upload</UploadId>"
                    b"</InitiateMultipartUploadResult>",
                )
            else:
                if state["fail_complete"]:
                    self.respond(
                        400,
                        b"<Error><Code>InvalidPart</Code>"
                        b"<Message>https://private.example/secret-key?token=secret</Message>"
                        b"</Error>",
                    )
                    return
                root = ET.fromstring(body)
                numbers = [
                    int(node.text)
                    for node in root.iter()
                    if node.tag.rsplit("}", 1)[-1] == "PartNumber"
                ]
                state["objects"][key] = b"".join(state["parts"][n] for n in numbers)
                state["completed"] += 1
                state["parts"].clear()
                self.respond(
                    200,
                    b'<CompleteMultipartUploadResult><ETag>"verified-object"</ETag>'
                    b"</CompleteMultipartUploadResult>",
                )

        def do_PUT(self):
            key, query = self.parse()
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            if len(body) != length:
                # A progress deadline can close a request body mid-part.
                self.close_connection = True
                return
            if "partNumber" in query:
                number = int(query["partNumber"][0])
                if state["fail_parts"]:
                    self.respond(500, b"<Error><Code>InternalError</Code></Error>")
                    return
                with lock:
                    state["parts"][number] = body
                    state["part_numbers"].append(number)
                self.respond(200, etag='"part-etag"')
            else:
                state["objects"][key] = body
                self.respond(200, etag='"verified-object"')

        def do_DELETE(self):
            state["aborted"] += 1
            state["parts"].clear()
            self.respond(204)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        region_name="us-east-1",
        aws_access_key_id="fixture-access",
        aws_secret_access_key="fixture-secret",
        config=Config(
            signature_version="s3v4",
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 0},
            max_pool_connections=16,
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    try:
        yield client, state
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "fixture.ckpt"
    # Sparse file avoids consuming scarce local disk; production 8 MiB parts
    # still require three real HTTP requests for this 17 MiB payload.
    with path.open("wb") as handle:
        handle.truncate(17 * 1024**2)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    try:
        yield path, digest.hexdigest()
    finally:
        path.unlink(missing_ok=True)


def test_real_sdk_uploads_multipart_and_verifies_all_bytes(gateway, checkpoint, capsys):
    client, state = gateway
    path, digest = checkpoint
    store = ManagedCheckpointStore(client, "checkpoint-fixture", lambda: None)
    receipt = store.put_file("step.ckpt", path, digest)
    assert state["created"] == state["completed"] == 1
    assert sorted(state["part_numbers"]) == [1, 2, 3]
    assert state["aborted"] == 0
    assert state["readbacks"] == 1
    assert receipt["sha256"] == digest
    assert receipt["bytes"] == path.stat().st_size
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["phase"] for event in events] == ["started", "upload", "verify", "completed"]
    assert events[-1]["bytes"] == receipt["bytes"]
    assert "fixture-secret" not in json.dumps(events)


def test_existing_checkpoint_is_verified_without_upload(gateway, tmp_path):
    client, state = gateway
    payload = b"an existing immutable checkpoint"
    path = tmp_path / "existing.ckpt"
    path.write_bytes(payload)
    state["objects"]["/checkpoint-fixture/step.ckpt"] = payload
    store = ManagedCheckpointStore(client, "checkpoint-fixture", lambda: None)
    store.put_file("step.ckpt", path, hashlib.sha256(payload).hexdigest())
    assert state["created"] == state["completed"] == 0
    assert state["readbacks"] == 1
    assert path.read_bytes() == payload


def test_corrupt_complete_remote_object_fails_and_retains_local_file(gateway, checkpoint):
    client, state = gateway
    path, digest = checkpoint
    state["corrupt_readback"] = True
    store = ManagedCheckpointStore(client, "checkpoint-fixture", lambda: None)
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.put_file("step.ckpt", path, digest)
    assert path.exists()
    assert state["completed"] == 1


def test_deadline_in_sdk_body_callback_aborts_multipart(gateway, checkpoint):
    client, state = gateway
    path, digest = checkpoint
    caller = threading.current_thread()

    def check():
        if threading.current_thread() is not caller:
            raise TimeoutError("private endpoint must not appear in failure")

    store = ManagedCheckpointStore(client, "checkpoint-fixture", check)
    with pytest.raises(TimeoutError, match="^Checkpoint upload exceeded its reserved time$"):
        store.put_file("step.ckpt", path, digest)
    assert state["created"] == state["aborted"] == 1
    assert state["completed"] == 0
    assert not state["objects"]
    assert path.exists()


def test_sdk_part_failure_aborts_and_exposes_only_safe_error(gateway, checkpoint, capsys):
    client, state = gateway
    path, digest = checkpoint
    state["fail_parts"] = True
    store = ManagedCheckpointStore(client, "checkpoint-fixture", lambda: None)
    with pytest.raises(RuntimeError) as failure:
        store.put_file("step.ckpt", path, digest)
    assert str(failure.value) == "Managed checkpoint upload failed (S3UploadFailedError)"
    assert state["created"] == state["aborted"] == 1
    assert state["completed"] == 0
    assert not state["objects"]
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["event"] == "checkpoint_transfer_failed"
    assert event["transfer_stage"] == "upload"
    assert event["last_transfer_operation"] == "UploadPart"
    assert event["operation_name"] == "UploadPart"
    assert event["error_code"] == "InternalError"
    assert event["http_status"] == 500
    assert event["exception_chain"] == ["RuntimeError", "S3UploadFailedError", "ClientError"]
    assert event["operation_trace"][-1] == "AbortMultipartUpload"


def test_completion_failure_diagnostics_preserve_operation_after_abort(gateway, checkpoint, capsys):
    client, state = gateway
    path, digest = checkpoint
    state["fail_complete"] = True
    store = ManagedCheckpointStore(client, "checkpoint-fixture", lambda: None)
    with pytest.raises(RuntimeError):
        store.put_file("step.ckpt", path, digest)
    assert state["created"] == state["aborted"] == 1
    assert state["completed"] == 0
    output = capsys.readouterr().out
    event = json.loads(output.splitlines()[-1])
    assert event["operation_name"] == "CompleteMultipartUpload"
    assert event["last_transfer_operation"] == "CompleteMultipartUpload"
    assert event["error_code"] == "InvalidPart"
    assert event["http_status"] == 400
    assert event["operation_trace"][-2:] == ["CompleteMultipartUpload", "AbortMultipartUpload"]
    assert event["bytes"] == path.stat().st_size
    for private in ("https://", "private.example", "secret-key", "token=secret", "step.ckpt"):
        assert private not in output


def test_safe_exception_details_traverse_cause_and_context_without_messages():
    private = "https://private.example/secret-key?token=secret"
    sdk_error = ClientError(
        {
            "Error": {"Code": "SlowDown", "Message": private, "Key": private},
            "ResponseMetadata": {
                "HTTPStatusCode": 503,
                "RequestId": private,
                "HTTPHeaders": {"authorization": private},
            },
        },
        "CompleteMultipartUpload",
    )
    wrapper = RuntimeError(private)
    wrapper.__context__ = sdk_error
    cause = ValueError(private)
    wrapper.__cause__ = cause
    # Cycles can appear in hand-built or re-raised exception chains.
    cause.__context__ = wrapper
    details = safe_exception_details(wrapper)
    assert details["exception_chain"] == ["RuntimeError", "ValueError", "ClientError"]
    assert details["sdk_errors"] == [
        {"operation_name": "CompleteMultipartUpload", "error_code": "SlowDown", "http_status": 503}
    ]
    assert private not in json.dumps(details)


def test_safe_exception_details_reject_unstructured_sdk_fields():
    private = "https://private.example/secret-key?token=secret"
    error = ClientError(
        {
            "Error": {"Code": private, "Message": private},
            "ResponseMetadata": {"HTTPStatusCode": private},
        },
        private,
    )
    assert safe_exception_details(error) == {"exception_chain": ["ClientError"], "sdk_errors": []}


def test_progress_serializes_worker_updates_and_retry_corrections(capsys):
    progress = _UploadProgress(10_000, lambda: None)
    threads = [threading.Thread(target=lambda: [progress(1) for _ in range(100)]) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    progress(-100)
    progress.phase("upload")
    event = json.loads(capsys.readouterr().out)
    assert event["bytes"] == 700
    assert event["rate_bytes_per_second"] >= 0

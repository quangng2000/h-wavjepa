"""Managed checkpoint uploads and process-bounded verified object downloads.

The official SDK owns multipart creation, retries, completion and abort cleanup.
Publication still requires the Store's complete remote SHA256 readback. This
adapter leaves the AudioSet/Nat staging transport and small JSON writes unchanged.
Training shard and checkpoint-resume downloads use killable SDK subprocesses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)

from src.stage_audioset import MAX_OBJECT_BYTES, StageLimit, Store

CHECKPOINT_TRANSFER_CONCURRENCY = 8
CHECKPOINT_PART_BYTES = 8 * 1024**2
DOWNLOAD_PART_BYTES = 16 * 1024**2
DOWNLOAD_CONCURRENCY = 4
_TRANSIENT_CODES = frozenset(
    {
        "InternalError",
        "InternalFailure",
        "RequestTimeout",
        "RequestTimeoutException",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)
_PERMANENT_CODES = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "AuthorizationHeaderMalformed",
        "BadDigest",
        "EntityTooSmall",
        "ExpiredToken",
        "InvalidAccessKeyId",
        "InvalidDigest",
        "InvalidPart",
        "InvalidPartOrder",
        "InvalidRequest",
        "InvalidToken",
        "NoSuchBucket",
        "NoSuchUpload",
        "NotImplemented",
        "SignatureDoesNotMatch",
    }
)
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_S3_OPERATIONS = frozenset(
    {
        "HeadObject",
        "GetObject",
        "PutObject",
        "CreateMultipartUpload",
        "UploadPart",
        "CompleteMultipartUpload",
        "AbortMultipartUpload",
    }
)


class _CheckpointDeadlineExceeded(TimeoutError):
    """Distinguish the caller's total deadline from an SDK socket timeout."""


def _exception_chain(error):
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)


def _retryable_transfer_error(error):
    """Retry only recognized transient failures, never integrity/deadline errors."""
    chain = list(_exception_chain(error))
    if any(
        isinstance(
            item, (_CheckpointDeadlineExceeded, ValueError, TypeError, KeyError, AssertionError)
        )
        for item in chain
    ):
        return False
    retryable = False
    for item in chain:
        if isinstance(item, ClientError):
            code = item.response.get("Error", {}).get("Code")
            status = item.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in _PERMANENT_CODES or status in {401, 403, 501, 505}:
                return False
            retryable |= code in _TRANSIENT_CODES or (
                type(status) is int and (status in {408, 429} or 500 <= status <= 599)
            )
        elif isinstance(
            item,
            (
                ConnectTimeoutError,
                ReadTimeoutError,
                EndpointConnectionError,
                ConnectionClosedError,
                IncompleteReadError,
                ResponseStreamingError,
            ),
        ):
            retryable = True
    return retryable


def safe_exception_details(error):
    """Read only structured fields; SDK wrapper messages can contain secrets."""
    chain, sdk_errors = [], []
    for current in _exception_chain(error):
        name = type(current).__name__
        chain.append(name if _SAFE_TOKEN.fullmatch(name) else "UnknownException")
        operation = getattr(current, "operation_name", None)
        response = getattr(current, "response", None)
        if isinstance(response, dict):
            detail = {}
            if isinstance(operation, str) and operation in _S3_OPERATIONS:
                detail["operation_name"] = operation
            error_fields = response.get("Error")
            code = error_fields.get("Code") if isinstance(error_fields, dict) else None
            if isinstance(code, str) and _SAFE_TOKEN.fullmatch(code):
                detail["error_code"] = code
            metadata = response.get("ResponseMetadata")
            status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
            if type(status) is int and 100 <= status <= 599:
                detail["http_status"] = status
            if detail:
                sdk_errors.append(detail)
    return {"exception_chain": chain, "sdk_errors": sdk_errors}


class _TransferDiagnostics:
    """Keep a small operation-only trace without request or response payloads."""

    def __init__(self, client, progress):
        self.events = getattr(getattr(client, "meta", None), "events", None)
        self.progress = progress
        self.identifier = f"hwavjepa-checkpoint-{id(self)}"
        self.operations = []
        self.last_transfer_operation = None
        self.lock = threading.Lock()

    def before_call(self, model, **kwargs):
        operation = getattr(model, "name", None)
        if not isinstance(operation, str) or operation not in _S3_OPERATIONS:
            return
        with self.lock:
            if not self.operations or self.operations[-1] != operation:
                self.operations.append(operation)
                del self.operations[:-8]
            if operation != "AbortMultipartUpload":
                self.last_transfer_operation = operation

    def __enter__(self):
        if self.events is not None:
            self.events.register("before-call.s3.*", self.before_call, unique_id=self.identifier)
        return self

    def __exit__(self, exc_type, error, traceback):
        if self.events is not None:
            self.events.unregister("before-call.s3.*", unique_id=self.identifier)
        if error is not None:
            self.report_failure(error)
        return False

    def report_failure(self, error, *, will_retry=False):
        with self.progress.lock, self.lock:
            details = safe_exception_details(error)
            event = {
                "event": "checkpoint_transfer_failed",
                "transfer_stage": self.progress.current_phase,
                "attempt": self.progress.attempt,
                "will_retry": will_retry,
                "bytes": self.progress.bytes,
                "total_bytes": self.progress.total_bytes,
                "last_transfer_operation": self.last_transfer_operation,
                "operation_trace": list(self.operations),
                **details,
            }
            # Also expose the first structured SDK error at the top level
            # for compact logs, preserving all errors in the bounded list.
            if details["sdk_errors"]:
                event.update(details["sdk_errors"][0])
            print(json.dumps(event, allow_nan=False), flush=True)


class _UploadProgress:
    """Serialize SDK worker callbacks, including retry byte-count corrections."""

    def __init__(self, total_bytes, check):
        self.total_bytes = total_bytes
        self.check = check
        self.bytes = 0
        self.started = self.last_report = time.monotonic()
        self.current_phase = "started"
        self.attempt = 1
        self.lock = threading.Lock()

    def _report(self, phase, now):
        elapsed = max(0.0, now - self.started)
        print(
            json.dumps(
                {
                    "event": "checkpoint_transfer",
                    "phase": phase,
                    "attempt": self.attempt,
                    "bytes": self.bytes,
                    "total_bytes": self.total_bytes,
                    "elapsed_seconds": round(elapsed, 3),
                    "rate_bytes_per_second": round(self.bytes / elapsed, 1) if elapsed else 0,
                },
                allow_nan=False,
            ),
            flush=True,
        )
        self.last_report = now

    def phase(self, phase):
        with self.lock:
            self.current_phase = phase
            self.check()
            self._report(phase, time.monotonic())

    def __call__(self, bytes_transferred):
        with self.lock:
            # s3transfer invokes this while request bodies are read. Raising
            # fails the transfer and runs its registered AbortMultipartUpload.
            # Socket timeouts still bound a blocked in-flight request.
            self.check()
            self.bytes = max(0, self.bytes + bytes_transferred)
            now = time.monotonic()
            if now - self.last_report >= 10:
                self._report("upload", now)


class ManagedCheckpointStore(Store):
    """Upload local checkpoints using the SDK, then verify the complete object."""

    def __init__(self, client, bucket, check):
        def check_deadline():
            try:
                check()
            except TimeoutError as error:
                raise _CheckpointDeadlineExceeded(
                    "Checkpoint publication exceeded its reserved time"
                ) from error

        super().__init__(client, bucket, check_deadline)

    def put_file(self, key, path: Path, digest):
        size = path.stat().st_size
        if size > MAX_OBJECT_BYTES:
            raise StageLimit("Checkpoint exceeds the supported 4 GiB object limit")
        progress = _UploadProgress(size, self.check)
        with _TransferDiagnostics(self.client, progress) as diagnostics:
            progress.phase("started")
            for attempt in (1, 2):
                progress.attempt = attempt
                try:
                    if attempt > 1:
                        # The SDK has already joined its workers and attempted
                        # abort cleanup. Completion may nevertheless have landed:
                        # HEAD first, then prove the entire object before reuse.
                        progress.phase("reconcile")
                    if self.head(key) is None:
                        progress.bytes = 0
                        progress.phase("upload")
                        self._upload_file(key, path, progress)
                    progress.phase("verify")
                    receipt = self.verify(key, digest, size)
                    progress.bytes = size
                    progress.phase("completed")
                    return receipt
                except Exception as error:
                    # The original caller's deadline covers both attempts and
                    # readbacks; no retry creates a fresh time allowance.
                    self.check()
                    if attempt == 2 or not _retryable_transfer_error(error):
                        raise
                    diagnostics.report_failure(error, will_retry=True)

    def _upload_file(self, key, path, progress):
        config = TransferConfig(
            multipart_threshold=CHECKPOINT_PART_BYTES,
            multipart_chunksize=CHECKPOINT_PART_BYTES,
            max_concurrency=CHECKPOINT_TRANSFER_CONCURRENCY,
            use_threads=True,
            preferred_transfer_client="classic",
        )
        try:
            self.client.upload_file(
                Filename=str(path),
                Bucket=self.bucket,
                Key=key,
                Config=config,
                Callback=progress,
            )
        except TimeoutError:
            raise TimeoutError("Checkpoint upload exceeded its reserved time") from None
        except Exception as error:
            # SDK messages can carry endpoints, signed URLs or local paths.
            raise RuntimeError(
                f"Managed checkpoint upload failed ({type(error).__name__})"
            ) from None


def managed_download_client(client):
    """Real SDK clients always use process isolation; injected memory stores do not."""
    return isinstance(client, BaseClient)


def _download_hash(path, check):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024**2):
            check()
            digest.update(chunk)
    check()
    return digest.hexdigest()


def _stop_download(process, grace):
    """Kill SDK threads with their process, never join an unbounded SDK executor."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=grace)
    else:
        process.wait()


def download_verified_object(
    client,
    bucket,
    key,
    destination,
    *,
    size,
    sha256,
    timeout_seconds,
    credential_scope,
    check=lambda: None,
):
    """Bound official SDK parallel GETs, hash verification and child cleanup.

    Popen works inside daemon DataLoader workers, unlike multiprocessing.Process.
    The subprocess owns all SDK threads and a private scratch directory. Nothing
    becomes a cache hit or resume file until its complete SHA256 is verified.
    Credentials are inherited from the existing scoped environment, never passed
    in arguments, a config file, a result, or logs.
    """
    if not managed_download_client(client):
        raise TypeError("Managed downloads require a real botocore client")
    if (
        type(size) is not int
        or not 0 < size <= MAX_OBJECT_BYTES
        or not isinstance(sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", sha256) is None
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or credential_scope not in {"audio", "checkpoint"}
    ):
        raise ValueError("Invalid verified-download bounds")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Verified download destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    grace = min(2.0, timeout_seconds / 4)
    deadline = started + timeout_seconds - 2 * grace

    def bounded_check():
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError("Verified object download exceeded its total deadline")

    bounded_check()
    scratch = Path(tempfile.mkdtemp(prefix=".verified-download-", dir=destination.parent))
    temporary = scratch / "object.partial"
    request = {
        "endpoint": client.meta.endpoint_url,
        "region": client.meta.region_name,
        "bucket": bucket,
        "key": key,
        "path": str(temporary.resolve()),
        "size": size,
        "sha256": sha256,
        "credential_scope": credential_scope,
        "parent_pid": os.getpid(),
    }
    process = None
    try:
        print(
            json.dumps(
                {
                    "event": "verified_object_download",
                    "phase": "launch",
                    "credential_scope": credential_scope,
                    "rank": os.environ.get("RANK"),
                    "sha256": sha256,
                    "total_bytes": size,
                    "timeout_seconds": timeout_seconds,
                }
            ),
            flush=True,
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.checkpoint_transfer",
                "--download",
                json.dumps(request),
            ],
            cwd=Path(__file__).resolve().parents[1],
            start_new_session=True,
        )
        while process.poll() is None:
            bounded_check()
            try:
                process.wait(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        bounded_check()
        if process.returncode != 0:
            raise RuntimeError(
                "Managed object download failed; see structured transfer diagnostics"
            )
        if (
            not temporary.is_file()
            or temporary.is_symlink()
            or temporary.stat().st_size != size
            or _download_hash(temporary, bounded_check) != sha256
        ):
            raise ValueError("Managed object download size or SHA256 mismatch")
        bounded_check()
        temporary.replace(destination)
    finally:
        # Never remove files that a still-running SDK thread might be writing.
        if process is not None:
            _stop_download(process, grace)
        shutil.rmtree(scratch)


def _download_in_child(request):
    # DataLoader workers may be terminated without unwinding Python finally
    # blocks. Their SDK child must not outlive them on a persistent GPU node.
    parent_pid = request["parent_pid"]
    if sys.platform == "linux":
        import ctypes

        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise RuntimeError("Could not arm transfer parent-death protection")
    else:

        def watch_parent():
            while os.getppid() == parent_pid:
                time.sleep(0.1)
            os._exit(1)

        threading.Thread(target=watch_parent, daemon=True).start()
    if os.getppid() != parent_pid:
        raise RuntimeError("Download parent exited before child startup")

    import boto3
    from botocore.config import Config

    endpoint = urlparse(request["endpoint"])
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path not in {"", "/"}
        or (
            endpoint.scheme == "http" and endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        raise ValueError("Invalid object-storage endpoint")
    credentials = {}
    scope = request["credential_scope"]
    if scope == "checkpoint":
        access = os.environ.get("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID")
        secret = os.environ.get("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY")
        if not access or not secret:
            raise ValueError("Checkpoint download requires its separate credential pair")
        credentials = {
            "aws_access_key_id": access,
            "aws_secret_access_key": secret,
            "aws_session_token": os.environ.get("HWAVJEPA_CHECKPOINT_SESSION_TOKEN"),
        }
    elif scope != "audio":
        raise ValueError("Unknown download credential scope")
    client = boto3.client(
        "s3",
        endpoint_url=request["endpoint"],
        region_name=request["region"],
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=30,
            retries={"mode": "standard", "max_attempts": 2},
            max_pool_connections=8,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
        **credentials,
    )
    size = request["size"]
    head = client.head_object(Bucket=request["bucket"], Key=request["key"])
    if head.get("ContentLength") != size:
        raise ValueError("Object size differs from its verified descriptor")
    started = last_report = time.monotonic()
    transferred = 0
    lock = threading.Lock()

    def report(phase):
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "event": "verified_object_download",
                    "phase": phase,
                    "credential_scope": scope,
                    "rank": os.environ.get("RANK"),
                    "pid": os.getpid(),
                    "sha256": request["sha256"],
                    "bytes": transferred,
                    "total_bytes": size,
                    "elapsed_seconds": round(elapsed, 3),
                    "rate_bytes_per_second": round(transferred / elapsed, 1) if elapsed else 0,
                }
            ),
            flush=True,
        )

    def progress(count):
        nonlocal transferred, last_report
        with lock:
            transferred = max(0, transferred + count)
            if time.monotonic() - last_report >= 10:
                report("download")
                last_report = time.monotonic()

    report("started")
    path = Path(request["path"])
    client.download_file(
        Bucket=request["bucket"],
        Key=request["key"],
        Filename=str(path),
        Config=TransferConfig(
            multipart_threshold=DOWNLOAD_PART_BYTES,
            multipart_chunksize=DOWNLOAD_PART_BYTES,
            max_concurrency=DOWNLOAD_CONCURRENCY,
            num_download_attempts=3,
            use_threads=True,
            preferred_transfer_client="classic",
        ),
        Callback=progress,
    )
    report("verify")
    if path.stat().st_size != size or _download_hash(path, lambda: None) != request["sha256"]:
        raise ValueError("Downloaded object size or SHA256 mismatch")
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    report("completed")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "--download":
            raise ValueError("Expected the internal verified-download entry point")
        _download_in_child(json.loads(sys.argv[2]))
    except Exception as error:
        print(
            json.dumps(
                {"event": "verified_object_download_failed", **safe_exception_details(error)}
            ),
            flush=True,
        )
        raise SystemExit(1) from None

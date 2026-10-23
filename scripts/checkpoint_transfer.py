"""Managed multipart checkpoint transfers with bounded concurrency and progress.

The official SDK owns multipart creation, retries, completion and abort cleanup.
Publication still requires the Store's complete remote SHA256 readback. This
adapter leaves the AudioSet/Nat staging transport and small JSON writes unchanged.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from boto3.s3.transfer import TransferConfig
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

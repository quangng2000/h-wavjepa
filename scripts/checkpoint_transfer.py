"""Managed multipart checkpoint transfers with bounded concurrency and progress.

The official SDK owns multipart creation, retries, completion and abort cleanup.
Publication still requires the Store's complete remote SHA256 readback. This
adapter leaves the AudioSet/Nat staging transport and small JSON writes unchanged.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from boto3.s3.transfer import TransferConfig

from src.stage_audioset import MAX_OBJECT_BYTES, StageLimit, Store

CHECKPOINT_TRANSFER_CONCURRENCY = 8
CHECKPOINT_PART_BYTES = 8 * 1024**2


class _UploadProgress:
    """Serialize SDK worker callbacks, including retry byte-count corrections."""

    def __init__(self, total_bytes, check):
        self.total_bytes = total_bytes
        self.check = check
        self.bytes = 0
        self.started = self.last_report = time.monotonic()
        self.lock = threading.Lock()

    def _report(self, phase, now):
        elapsed = max(0.0, now - self.started)
        print(
            json.dumps(
                {
                    "event": "checkpoint_transfer",
                    "phase": phase,
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

    def put_file(self, key, path: Path, digest):
        size = path.stat().st_size
        if size > MAX_OBJECT_BYTES:
            raise StageLimit("Checkpoint exceeds the supported 4 GiB object limit")
        progress = _UploadProgress(size, self.check)
        progress.phase("started")
        if self.head(key) is None:
            progress.phase("upload")
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
        # Content-addressed existing objects are verified, never overwritten.
        progress.phase("verify")
        receipt = self.verify(key, digest, size)
        progress.bytes = size
        progress.phase("completed")
        return receipt

"""End the clean curriculum phase when the pinned Nat asset receipt is ready."""

from __future__ import annotations

import hashlib
import json

from lightning.pytorch.callbacks import Callback

from scripts.start_nat_segment import (
    INVENTORY,
    MAX_RECEIPT_BYTES,
    _bounded_get,
    _endpoint,
    _metadata_deadline,
    _safe_key,
    _validate_ready,
    metadata_client,
)


class NatReadyStop(Callback):
    """Poll only metadata on rank zero and stop all ranks at the same boundary.

    The segment runner still performs validation and publishes its final full
    checkpoint. The next CPU gate verifies the actual manifest before Nat can
    launch. Missing, malformed or temporarily inaccessible receipts leave the
    bounded clean phase running; no asset or checkpoint is modified here.
    """

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        ready_key: str,
        poll_every_steps: int = 100,
        metadata_timeout_seconds: float = 20,
    ):
        super().__init__()
        self.endpoint = _endpoint(endpoint)
        if not bucket or not _safe_key(ready_key):
            raise ValueError("A bucket and safe Nat readiness key are required")
        if type(poll_every_steps) is not int or poll_every_steps < 1:
            raise ValueError("Readiness polling interval must be a positive integer")
        if not 0 < metadata_timeout_seconds <= 30:
            raise ValueError("Readiness metadata timeout must be at most 30 seconds")
        self.bucket, self.ready_key = bucket, ready_key
        self.poll_every_steps = poll_every_steps
        self.metadata_timeout_seconds = metadata_timeout_seconds
        self.last_step = -1
        self.client = None

    def _is_ready(self):
        try:
            with _metadata_deadline(self.metadata_timeout_seconds):
                if self.client is None:
                    self.client = metadata_client(self.endpoint)
                inventory_bytes = INVENTORY.read_bytes()
                payload = _bounded_get(self.client, self.bucket, self.ready_key, MAX_RECEIPT_BYTES)
                _validate_ready(
                    json.loads(payload),
                    bucket=self.bucket,
                    endpoint=self.endpoint,
                    ready_key=self.ready_key,
                    inventory=json.loads(inventory_bytes),
                    inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                )
                return True
        except Exception:
            # SDK exceptions can contain signed URLs. A transient failure must
            # neither leak credentials nor interrupt valid clean training.
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step == self.last_step or step % self.poll_every_steps:
            return
        self.last_step = step
        ready = self._is_ready() if trainer.is_global_zero else False
        ready = trainer.strategy.broadcast(ready, src=0)
        if ready:
            trainer.should_stop = True
            if trainer.is_global_zero:
                print(
                    json.dumps({"curriculum": "nat_assets_ready", "clean_stop_step": step}),
                    flush=True,
                )

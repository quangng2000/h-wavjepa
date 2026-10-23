"""Request a normal checkpointed train end without terminating the GPU process.

An operator atomically writes the exact run/segment/nonce identity from the
current segment plan to its local request path. Rank zero reads this small file
at completed optimizer steps and broadcasts the decision. Lightning's normal
validation and final durable checkpoint lifecycle remains responsible for exit.
This hook must be installed before training starts; it cannot retrofit a live run.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from lightning.pytorch.callbacks import Callback

MAX_REQUEST_BYTES = 4096


class OperatorStop(Callback):
    """Stop all ranks at one batch boundary, accepting only this launch's marker."""

    def __init__(self, request_path: str, run_id: str, segment_id: str, nonce: str):
        path = Path(request_path)
        if not path.is_absolute():
            raise ValueError("Operator stop request path must be absolute")
        for value in (run_id, segment_id):
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
            ):
                raise ValueError("Operator stop needs explicit run and segment identities")
        if not isinstance(nonce, str) or re.fullmatch(r"[a-f0-9]{32}", nonce) is None:
            raise ValueError("Operator stop needs a fresh 32-character hexadecimal launch nonce")
        self.request_path = path
        self.expected = {
            "schema_version": 1,
            "run_id": run_id,
            "segment_id": segment_id,
            "nonce": nonce,
        }
        self.last_step = -1
        self.stop_requested = False
        self._rejection_reported = False
        self._stop_reported = False

    def _is_requested(self) -> bool:
        try:
            # NOFOLLOW rejects symlinks; NONBLOCK prevents an accidental FIFO from
            # blocking a collective. Only a small regular file is ever parsed.
            descriptor = os.open(self.request_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as handle:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request file")
                payload = handle.read(MAX_REQUEST_BYTES + 1)
            if len(payload) > MAX_REQUEST_BYTES:
                raise ValueError("Oversized request")
            request = json.loads(payload)
            if (
                not isinstance(request, dict)
                or type(request.get("schema_version")) is not int
                or request != self.expected
            ):
                raise ValueError("Request does not match this launch")
            return True
        except FileNotFoundError:
            self._rejection_reported = False
            return False
        except (OSError, ValueError, UnicodeError, RecursionError):
            if not self._rejection_reported:
                print(json.dumps({"event": "operator_stop_request_rejected"}), flush=True)
                self._rejection_reported = True
            return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step == self.last_step:
            return
        self.last_step = step
        requested = self._is_requested() if trainer.is_global_zero else False
        requested = trainer.strategy.broadcast(requested, src=0)
        if requested:
            self.stop_requested = True
            trainer.should_stop = True
            if trainer.is_global_zero and not self._stop_reported:
                print(
                    json.dumps(
                        {
                            "event": "operator_stop_requested",
                            "step": step,
                            "run_id": self.expected["run_id"],
                            "segment_id": self.expected["segment_id"],
                        }
                    ),
                    flush=True,
                )
                self._stop_reported = True

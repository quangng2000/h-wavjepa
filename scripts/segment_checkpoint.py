"""Verified private-S3 checkpoints for explicitly bounded training segments.

No cloud resources or credentials are created here. The caller must provide an
exclusive, bucket-scoped writer for one owned prefix. S3 conditional writes and
user metadata are deliberately not used: this gateway does not enforce them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

from lightning.pytorch.callbacks import Timer
from lightning.pytorch.strategies import DDPStrategy

from src.callbacks.step_checkpoint import StepCheckpoint
from src.stage_audioset import MAX_OBJECT_BYTES, Store
from src.utils.runtime import atomic_json, file_sha256

SHA = re.compile(r"[a-f0-9]{64}")
RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{7,79}")


def minimum_checkpoint_collective_timeout(transfer_timeout_seconds: float) -> int:
    """Allow publication's cooperative deadline plus SDK/collective completion time."""
    if not math.isfinite(transfer_timeout_seconds) or transfer_timeout_seconds <= 0:
        raise ValueError("Checkpoint transfer timeout must be finite and positive")
    return math.ceil(transfer_timeout_seconds + 120)


def checkpoint_client(endpoint: str):
    """Use a separate runtime environment credential pair; never log it."""
    import boto3
    from botocore.config import Config

    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Checkpoint endpoint must be an HTTPS origin without credentials")
    access = os.environ.get("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID")
    secret = os.environ.get("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise ValueError("Set the separate HWAVJEPA_CHECKPOINT_* credential pair at launch")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("HWAVJEPA_CHECKPOINT_REGION", "auto"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=os.environ.get("HWAVJEPA_CHECKPOINT_SESSION_TOKEN"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 2},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            max_pool_connections=16,
        ),
    )


class CheckpointRepository:
    """Keep at most two checkpoint objects beneath an explicitly owned prefix.

    Before writing a replacement, remove only older owned checkpoint objects,
    retaining the current latest. Publish latest.json only after complete remote
    SHA256 readback. A failed write leaves the preceding checkpoint intact and
    raises; it never silently falls back or declares a successful handoff.
    """

    def __init__(
        self,
        client,
        bucket: str,
        prefix: str,
        run_id: str,
        index_sha256: str,
        check,
        base_data_seed: int = 42,
        global_batch: int = 64,
        naturalistic_manifest_sha256: str | None = None,
        clean_binaural: bool = False,
        curriculum_source: dict | None = None,
        managed_upload: bool = False,
    ):
        if not RUN_ID.fullmatch(run_id) or not SHA.fullmatch(index_sha256):
            raise ValueError("Expected a stable run ID and frozen index SHA256")
        if type(base_data_seed) is not int or not 0 <= base_data_seed < 2**32 - 375000:
            raise ValueError("Expected a nonnegative bounded base data seed")
        if type(global_batch) is not int or global_batch < 1:
            raise ValueError("Expected a positive integer global recording batch")
        if naturalistic_manifest_sha256 is not None and (
            not isinstance(naturalistic_manifest_sha256, str)
            or not SHA.fullmatch(naturalistic_manifest_sha256)
            or global_batch != 16
        ):
            raise ValueError(
                "Nat checkpoint ownership requires a manifest SHA256 and global batch 16"
            )
        if clean_binaural and (naturalistic_manifest_sha256 is not None or global_batch != 16):
            raise ValueError("Clean binaural checkpoints require batch 16 and no Nat manifest")
        if curriculum_source is not None:
            if (
                naturalistic_manifest_sha256 is None
                or not isinstance(curriculum_source, dict)
                or set(curriculum_source) != {"run_id", "prefix", "checkpoint_sha256"}
                or not RUN_ID.fullmatch(curriculum_source.get("run_id", ""))
                or not SHA.fullmatch(curriculum_source.get("checkpoint_sha256", ""))
                or curriculum_source.get("prefix") == prefix
            ):
                raise ValueError("Curriculum lineage requires a distinct pinned clean source")
        parts = prefix.split("/")
        if (
            len(parts) < 2
            or any(not part or part in {".", ".."} for part in parts)
            or "\\" in prefix
            or parts[-1] != run_id
        ):
            raise ValueError("Owned prefix must have at least two components and end in run_id")
        if not bucket or "/" in bucket:
            raise ValueError("Expected a private checkpoint bucket name")
        if managed_upload:
            from scripts.checkpoint_transfer import ManagedCheckpointStore

            self.store = ManagedCheckpointStore(client, bucket, check)
        else:
            self.store = Store(client, bucket, check)
        self.prefix, self.run_id, self.index_sha256 = prefix, run_id, index_sha256
        self.base_data_seed = base_data_seed
        self.owner = {
            "schema": 1,
            "owner": "h-wavjepa-training-segments",
            "run_id": run_id,
            "index_sha256": index_sha256,
            "total_optimizer_steps": 375000,
            "global_batch": global_batch,
            "base_data_seed": base_data_seed,
            "data_seed_policy": "base_plus_resume_step",
        }
        if naturalistic_manifest_sha256 is not None or clean_binaural:
            self.owner.update(
                training_phase="clean_binaural" if clean_binaural else "naturalistic",
                naturalistic_manifest_sha256=naturalistic_manifest_sha256,
                input_channels=2,
                masked_crops_per_recording=8,
                masked_crop_samples=32160,
            )
        if curriculum_source is not None:
            self.owner["curriculum_source"] = dict(curriculum_source)
        self.handoff_key = f"{prefix}/clean-handoff.json"
        self.owner_key, self.latest_key = f"{prefix}/owner.json", f"{prefix}/latest.json"
        self.object_pattern = re.compile(
            re.escape(prefix) + r"/checkpoints/step_[0-9]{7}_[a-f0-9]{64}\.ckpt"
        )

    def _keys(self, prefix: str) -> list[str]:
        self.store.check()
        result = self.store.client.list_objects_v2(
            Bucket=self.store.bucket, Prefix=prefix, MaxKeys=20
        )
        if result.get("IsTruncated"):
            raise ValueError("Unexpected large checkpoint prefix; refusing adoption or cleanup")
        return [entry["Key"] for entry in result.get("Contents", [])]

    def assert_owner(self):
        if json.loads(self.store.read(self.owner_key)) != self.owner:
            raise ValueError("Checkpoint prefix belongs to a different run or frozen data index")

    def open(self, *, initialize: bool):
        if self.store.head(self.handoff_key) is not None:
            raise ValueError("Clean curriculum source is sealed after its completed handoff")
        if self.store.head(self.owner_key) is None:
            if not initialize or self._keys(self.prefix + "/"):
                raise ValueError("Cannot resume or adopt an unowned/nonempty checkpoint prefix")
            self.store.put_json(self.owner_key, self.owner, immutable=True)
        self.assert_owner()
        latest = self.latest()
        if initialize and latest is not None:
            raise ValueError("Existing checkpoint requires explicit resume")
        if not initialize and latest is None:
            raise ValueError("Resume requested but no verified latest receipt exists")
        return latest

    def _validate_record(self, record: dict):
        if (
            not isinstance(record, dict)
            or type(record.get("step")) is not int
            or not 1 <= record["step"] <= 375000
            or not isinstance(record.get("sha256"), str)
            or not SHA.fullmatch(record["sha256"])
            or type(record.get("bytes")) is not int
            or not 0 < record["bytes"] <= MAX_OBJECT_BYTES
            or record.get("key")
            != f"{self.prefix}/checkpoints/step_{record['step']:07d}_{record['sha256']}.ckpt"
        ):
            raise ValueError("Invalid or out-of-prefix checkpoint receipt")

    def latest(self):
        if self.store.head(self.latest_key) is None:
            return None
        receipt = json.loads(self.store.read(self.latest_key))
        if (
            not isinstance(receipt, dict)
            or receipt.get("owner") != self.owner
            or receipt.get("stream_positions_checkpointed") is not False
            or type(receipt.get("resolved_data_seed")) is not int
            or not self.base_data_seed
            <= receipt["resolved_data_seed"]
            <= self.base_data_seed + 375000
        ):
            raise ValueError("Invalid checkpoint lineage receipt")
        self._validate_record(receipt.get("checkpoint"))
        previous = receipt.get("previous")
        if previous is not None:
            self._validate_record(previous)
        return receipt

    def publish(self, path: Path, step: int, *, segment_id: str, data_seed: int) -> dict:
        self.assert_owner()
        if self.store.head(self.handoff_key) is not None:
            raise ValueError("Clean curriculum source is sealed after its completed handoff")
        if (
            type(data_seed) is not int
            or not self.base_data_seed <= data_seed < self.base_data_seed + step
        ):
            raise ValueError("Resolved data seed must equal base plus a preceding resume step")
        prior = self.latest()
        if prior and step <= prior["checkpoint"]["step"]:
            raise ValueError("A training segment must advance beyond its durable checkpoint")
        if prior:
            # Never prune the fallback if the current receipt's object vanished
            # or changed since publication. An unchanged verified ETag/size is
            # sufficient here; missing/different ETags trigger full SHA readback.
            self.store.confirm_receipt(prior["checkpoint"])
        digest = file_sha256(path)
        key = f"{self.prefix}/checkpoints/step_{step:07d}_{digest}.ckpt"
        self._validate_record(
            {"step": step, "key": key, "sha256": digest, "bytes": path.stat().st_size}
        )
        keep = {key}
        if prior:
            keep.add(prior["checkpoint"]["key"])
        keys = self._keys(self.prefix + "/checkpoints/")
        if any(not self.object_pattern.fullmatch(old) for old in keys):
            raise ValueError("Foreign object in owned checkpoint folder; refusing deletion")
        # Only the former latest must survive this transaction. Removing its
        # predecessor first bounds storage even if PUT/readback subsequently fails.
        for old in keys:
            if old not in keep:
                self.assert_owner()
                self.store.check()
                self.store.client.delete_object(Bucket=self.store.bucket, Key=old)
                if self.store.head(old) is not None:
                    raise RuntimeError("Owned checkpoint cleanup was not confirmed")
        record = self.store.put_file(key, path, digest) | {"step": step}
        receipt = {
            "owner": self.owner,
            "checkpoint": record,
            "previous": prior["checkpoint"] if prior else None,
            "segment_id": segment_id,
            "resolved_data_seed": data_seed,
            "stream_positions_checkpointed": False,
        }
        # This is not a compare-and-swap lock. Exactly one training writer must
        # own the prefix, as acknowledged by the launching orchestrator.
        self.store.put_json(self.latest_key, receipt)
        return receipt

    def publish_clean_handoff(self, receipt: dict, *, writer_finished: bool) -> dict:
        """Seal a completed clean phase, only after all training ranks have exited."""
        self.assert_owner()
        if (
            writer_finished is not True
            or self.owner.get("training_phase") != "clean_binaural"
            or receipt != self.latest()
            or receipt["checkpoint"]["step"] >= 375000
        ):
            raise ValueError("Clean handoff requires an exited writer and its advanced checkpoint")
        self.store.confirm_receipt(receipt["checkpoint"])
        handoff = {
            "schema_version": 1,
            "phase": "clean_binaural",
            "writer_finished": True,
            "ready_for_nat": True,
            "source_run_id": self.run_id,
            "source_prefix": self.prefix,
            "index_sha256": self.index_sha256,
            "global_batch": 16,
            "input_channels": 2,
            "masked_crops_per_recording": 8,
            "masked_crop_samples": 32160,
            "total_optimizer_steps": 375000,
            "checkpoint": receipt,
        }
        self.store.put_json(self.handoff_key, handoff, immutable=True)
        return handoff

    def clean_handoff(self, expected_checkpoint_sha256: str) -> dict:
        """Read a sealed source without adopting or reopening its writer."""
        self.assert_owner()
        handoff = json.loads(self.store.read(self.handoff_key))
        latest = self.latest()
        expected = {
            "schema_version": 1,
            "phase": "clean_binaural",
            "writer_finished": True,
            "ready_for_nat": True,
            "source_run_id": self.run_id,
            "source_prefix": self.prefix,
            "index_sha256": self.index_sha256,
            "global_batch": 16,
            "input_channels": 2,
            "masked_crops_per_recording": 8,
            "masked_crop_samples": 32160,
            "total_optimizer_steps": 375000,
            "checkpoint": latest,
        }
        if (
            self.owner.get("training_phase") != "clean_binaural"
            or latest is None
            or handoff != expected
            or latest["checkpoint"]["sha256"] != expected_checkpoint_sha256
            or latest["checkpoint"]["step"] >= 375000
        ):
            raise ValueError("Clean curriculum handoff does not match its pinned source checkpoint")
        return handoff

    def download_latest(self, destination: Path) -> dict:
        self.assert_owner()
        receipt = self.latest()
        if receipt is None:
            raise ValueError("No verified checkpoint to resume")
        record = receipt["checkpoint"]
        temporary = destination.with_suffix(".partial")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.store.client.get_object(Bucket=self.store.bucket, Key=record["key"])
        digest, count = hashlib.sha256(), 0
        try:
            with contextlib.closing(result["Body"]) as body, temporary.open("xb") as output:
                while chunk := body.read(1024**2):
                    self.store.check()
                    count += len(chunk)
                    if count > record["bytes"]:
                        raise ValueError("Checkpoint exceeds its verified receipt size")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("Checkpoint download SHA256 mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return receipt


class SegmentTimer(Timer):
    """Official Lightning Timer, with a fresh absolute deadline for each segment.

    Only this callback's elapsed time is reset on resume. Trainer loop, optimizer,
    scheduler and model/EMA counters retain their normal checkpoint restoration.
    """

    def __init__(self, training_deadline: float):
        if not math.isfinite(training_deadline):
            raise ValueError("Expected a finite segment deadline")
        self.training_deadline = training_deadline
        super().__init__(
            duration={"seconds": max(0.0, training_deadline - time.time())}, interval="step"
        )

    def load_state_dict(self, state_dict):
        # Deliberately ignore only the previous segment's Timer offset.
        self._offset = 0

    def on_fit_start(self, trainer, *args, **kwargs):
        self._duration = max(0.0, self.training_deadline - time.time())
        super().on_fit_start(trainer, *args, **kwargs)

    def on_train_start(self, trainer, pl_module):
        self._duration = max(0.0, self.training_deadline - time.time())
        super().on_train_start(trainer, pl_module)


class DurableStepCheckpoint(StepCheckpoint):
    """Save full local state, then verify private-S3 durability.

    Optional background publication overlaps storage I/O with training. At most
    one immutable step file is pending: periodic saves are skipped until it has
    finished, so Lightning cannot prune its upload source. All ranks poll the
    same short collective each batch; publication failures stop every rank.
    Final saves are drained before validation and successful fit completion.
    """

    def __init__(
        self,
        repository_options: dict,
        segment_id: str,
        data_seed: int,
        receipt_path: str,
        transfer_timeout_seconds: float,
        hard_deadline: float,
        first_checkpoint_step: int | None = None,
        async_upload: bool = False,
        **kwargs,
    ):
        if first_checkpoint_step is not None and (
            type(first_checkpoint_step) is not int or first_checkpoint_step < 1
        ):
            raise ValueError("first_checkpoint_step must be a positive integer or None")
        super().__init__(**kwargs)
        if type(async_upload) is not bool:
            raise ValueError("async_upload must be a boolean")
        minimum_checkpoint_collective_timeout(transfer_timeout_seconds)
        self.repository_options = dict(repository_options)
        self.segment_id, self.receipt_path = segment_id, Path(receipt_path)
        self.data_seed = data_seed
        self.transfer_timeout_seconds, self.hard_deadline = transfer_timeout_seconds, hard_deadline
        self.first_checkpoint_step = first_checkpoint_step
        self.last_receipt = None
        self.async_upload = async_upload
        self._executor = None
        self._publication = None
        self._pending_step = None

    def setup(self, trainer, pl_module, stage):
        if isinstance(trainer.strategy, DDPStrategy):
            # Lightning stores the timeout passed to init_process_group here.
            # Check the actual strategy, including when used outside our runner.
            timeout = trainer.strategy._timeout
            minimum = minimum_checkpoint_collective_timeout(self.transfer_timeout_seconds)
            if not isinstance(timeout, timedelta) or timeout.total_seconds() < minimum:
                raise ValueError(
                    f"Durable checkpoint DDP collective timeout must be at least {minimum}s "
                    "(checkpoint transfer timeout plus 120s); configure the strategy before fit"
                )
        super().setup(trainer, pl_module, stage)

    def _publish(self, path: Path, step: int):
        deadline = min(time.time() + self.transfer_timeout_seconds, self.hard_deadline - 30)

        def check():
            if time.time() >= deadline:
                raise TimeoutError("Checkpoint publication exceeded its reserved time")

        check()
        options = dict(self.repository_options)
        options.setdefault("managed_upload", self.async_upload)
        client = checkpoint_client(options.pop("endpoint"))
        repository = CheckpointRepository(client, **options, check=check)
        receipt = repository.publish(
            path, step, segment_id=self.segment_id, data_seed=self.data_seed
        )
        atomic_json(self.receipt_path, receipt)
        return receipt

    def _save_checkpoint(self, trainer, filepath):
        if self.async_upload and self._pending_step is not None:
            raise RuntimeError("Cannot replace an in-flight checkpoint")
        super()._save_checkpoint(trainer, filepath)
        outcome = None
        if trainer.is_global_zero:
            try:
                step = int(trainer.global_step)
                if self.async_upload:
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix="checkpoint-publisher"
                        )
                    # Only an already serialized file and scalar step cross the
                    # thread boundary; no live model, optimizer or trainer does.
                    self._publication = self._executor.submit(self._publish, Path(filepath), step)
                    outcome = {"pending_step": step}
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint_upload_started",
                                "step": step,
                                "bytes": Path(filepath).stat().st_size,
                            },
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                else:
                    outcome = {"receipt": self._publish(Path(filepath), step)}
            except Exception as error:
                # Avoid printing SDK exception messages: they can contain URLs.
                outcome = {"error": type(error).__name__}
        outcome = trainer.strategy.broadcast(outcome)
        self._accept_publication(trainer, outcome)

    def _accept_publication(self, trainer, outcome):
        if "error" in outcome:
            raise RuntimeError(
                f"Durable checkpoint publication failed ({outcome['error']}); stopping"
            )
        if "pending_step" in outcome:
            self._pending_step = outcome["pending_step"]
            return
        if "receipt" not in outcome:
            return
        self._publication = None
        self._pending_step = None
        self.last_receipt = outcome["receipt"]
        if trainer.is_global_zero:
            record = self.last_receipt["checkpoint"]
            print(
                json.dumps(
                    {
                        "event": "checkpoint_published",
                        "step": record["step"],
                        "sha256": record["sha256"],
                        "bytes": record["bytes"],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

    def _poll_publication(self, trainer):
        """One identical collective on all ranks, even when no upload is active."""
        outcome = None
        if trainer.is_global_zero:
            outcome = {}
            if self._publication is not None and self._publication.done():
                try:
                    outcome = {"receipt": self._publication.result()}
                except Exception as error:
                    outcome = {"error": type(error).__name__}
        self._accept_publication(trainer, trainer.strategy.broadcast(outcome))

    def _drain_publication(self, trainer):
        # Keep each collective short even at finalization. The worker's transfer
        # deadline bounds storage work; peers never sit inside one long broadcast.
        while self._pending_step is not None:
            self._poll_publication(trainer)
            if self._pending_step is not None:
                time.sleep(0.05)

    def _assert_final_receipt(self, trainer):
        if trainer.global_step > 0 and (
            self.last_receipt is None
            or self.last_receipt["checkpoint"]["step"] != trainer.global_step
        ):
            raise RuntimeError("Final optimizer step lacks a verified durable checkpoint")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        final_step = trainer.should_stop or (
            trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps
        )
        if self.async_upload:
            self._poll_publication(trainer)
            if final_step:
                self._drain_publication(trainer)
            elif self._pending_step is not None:
                # Backpressure is a skipped intermediate snapshot, never an
                # unbounded queue or deletion of bytes still being uploaded.
                return
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # Lightning may run full validation after Timer requests a stop. Make
        # this optimizer step durable BEFORE that tail consumes the reserve.
        # An optional early save protects the first shard refill without making
        # every later publication equally frequent. Normal cadence wins ties.
        if (
            (final_step or trainer.global_step == self.first_checkpoint_step)
            and trainer.global_step > 0
            and self._last_global_step_saved != trainer.global_step
        ):
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)
        if self.async_upload and final_step:
            self._drain_publication(trainer)
            self._assert_final_receipt(trainer)

    def on_train_end(self, trainer, pl_module):
        if self.async_upload:
            self._drain_publication(trainer)
        super().on_train_end(trainer, pl_module)
        if self.async_upload:
            self._drain_publication(trainer)
            self._assert_final_receipt(trainer)

    def on_fit_end(self, trainer, pl_module):
        if self.async_upload:
            self._drain_publication(trainer)
            self._assert_final_receipt(trainer)
        super().on_fit_end(trainer, pl_module)

    def teardown(self, trainer, pl_module, stage):
        # on_train_end/on_fit_end already propagated the verified outcome.
        self._shutdown_publisher()
        super().teardown(trainer, pl_module, stage)

    def on_exception(self, trainer, pl_module, exception):
        # Lightning skips callback.teardown on exceptions. Other ranks may have
        # exited, so drain only this process's I/O thread without any collective.
        self._shutdown_publisher()
        super().on_exception(trainer, pl_module, exception)

    def _shutdown_publisher(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

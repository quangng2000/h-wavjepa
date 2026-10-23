"""CPU-only gate between completed clean training and a Nat continuation.

Schedule this job with a dependency on successful clean training. It performs
bounded metadata polling, verifies the finished writer's checkpoint receipt and
the full Nat asset manifest, then publishes a small immutable handoff. It never
allocates GPUs, downloads checkpoint tensors, or imports the training stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.start_nat_segment import (  # noqa: E402
    INVENTORY,
    MAX_RECEIPT_BYTES,
    SHA256,
    _bounded_get,
    _endpoint,
    _metadata_deadline,
    _safe_key,
    _save_immutable,
    ensure_start_deadline,
    fetch_ready_manifest,
    metadata_client,
    utc_timestamp,
)

RECIPE = {
    "global_batch": 16,
    "input_channels": 2,
    "masked_crops_per_recording": 8,
    "masked_crop_samples": 32160,
    "total_optimizer_steps": 375000,
}


def checkpoint_metadata_client(endpoint):
    """Use the separately scoped checkpoint credentials without importing Lightning."""
    import boto3
    from botocore.config import Config

    access = os.environ.get("HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID")
    secret = os.environ.get("HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise ValueError("Separate checkpoint credentials are required")
    return boto3.client(
        "s3",
        endpoint_url=_endpoint(endpoint),
        region_name=os.environ.get("HWAVJEPA_CHECKPOINT_REGION", "auto"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        aws_session_token=os.environ.get("HWAVJEPA_CHECKPOINT_SESSION_TOKEN"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=20,
            retries={"mode": "standard", "total_max_attempts": 2},
            s3={"addressing_style": "path"},
        ),
    )


def _matches(value, expected):
    return isinstance(value, dict) and all(
        type(value.get(key)) is type(item) and value[key] == item for key, item in expected.items()
    )


def fetch_clean_handoff(client, *, bucket, prefix, run_id, index_sha256):
    if (
        not _safe_key(prefix)
        or not prefix.endswith("/" + run_id)
        or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{7,79}", run_id) is None
        or SHA256.fullmatch(index_sha256) is None
    ):
        raise ValueError("Invalid clean source identity")
    key = prefix + "/clean-handoff.json"
    payload = _bounded_get(client, bucket, key, MAX_RECEIPT_BYTES)
    handoff = json.loads(payload)
    if not _matches(
        handoff,
        RECIPE
        | {
            "schema_version": 1,
            "phase": "clean_binaural",
            "writer_finished": True,
            "ready_for_nat": True,
            "source_run_id": run_id,
            "source_prefix": prefix,
            "index_sha256": index_sha256,
        },
    ):
        raise ValueError("Clean handoff lacks a finished writer or matching curriculum recipe")
    latest = json.loads(_bounded_get(client, bucket, prefix + "/latest.json", MAX_RECEIPT_BYTES))
    owner = json.loads(_bounded_get(client, bucket, prefix + "/owner.json", MAX_RECEIPT_BYTES))
    if (
        not _matches(
            owner,
            RECIPE
            | {
                "schema": 1,
                "owner": "h-wavjepa-training-segments",
                "run_id": run_id,
                "index_sha256": index_sha256,
                "training_phase": "clean_binaural",
                "naturalistic_manifest_sha256": None,
                "data_seed_policy": "base_plus_resume_step",
            },
        )
        or handoff.get("checkpoint") != latest
        or latest.get("owner") != owner
        or latest.get("stream_positions_checkpointed") is not False
    ):
        raise ValueError("Clean handoff differs from the owned final checkpoint lineage")
    record = latest.get("checkpoint", {})
    step, digest, size = (record.get(field) for field in ("step", "sha256", "bytes"))
    if (
        type(step) is not int
        or not 0 < step < 375000
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or type(size) is not int
        or not 0 < size <= 5 * 1024**3
        or record.get("key") != f"{prefix}/checkpoints/step_{step:07d}_{digest}.ckpt"
        or type(latest.get("resolved_data_seed")) is not int
        or type(owner.get("base_data_seed")) is not int
        or not owner["base_data_seed"]
        <= latest["resolved_data_seed"]
        <= owner["base_data_seed"] + 375000
    ):
        raise ValueError("Clean checkpoint descriptor or remaining training steps are invalid")
    if client.head_object(Bucket=bucket, Key=record["key"]).get("ContentLength") != size:
        raise ValueError("Clean checkpoint object size differs from its verified receipt")
    return handoff, {
        "key": key,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def fetch_curriculum_handoff(
    client,
    *,
    bucket,
    ready_key,
    source_prefix,
    source_run_id,
    index_sha256,
    nat_bucket,
    nat_endpoint,
    nat_ready_key,
    nat_receipt,
):
    """Recheck the frozen gate handoff immediately before the GPU import."""
    if not _safe_key(ready_key):
        raise ValueError("Invalid curriculum handoff key")
    value = json.loads(_bounded_get(client, bucket, ready_key, MAX_RECEIPT_BYTES))
    if not _matches(
        value,
        {
            "schema_version": 1,
            "status": "complete",
            "ready_for_training": True,
            "source_prefix": source_prefix,
            "source_run_id": source_run_id,
            "index_sha256": index_sha256,
        },
    ):
        raise ValueError("Curriculum gate receipt does not match the expected clean run")
    handoff, descriptor = fetch_clean_handoff(
        client, bucket=bucket, prefix=source_prefix, run_id=source_run_id, index_sha256=index_sha256
    )
    record = handoff["checkpoint"]["checkpoint"]
    if (
        value.get("source_handoff") != descriptor
        or value.get("source_checkpoint_sha256") != record["sha256"]
        or value.get("source_optimizer_step") != record["step"]
        or value.get("nat")
        != {
            "bucket": nat_bucket,
            "endpoint": nat_endpoint,
            "ready_key": nat_ready_key,
            "manifest": nat_receipt["manifest"],
            "inventory_sha256": nat_receipt["inventory_sha256"],
        }
    ):
        raise ValueError("Curriculum checkpoint or Nat assets changed after CPU gate validation")
    return value


def _publish_immutable(client, bucket, key, value):
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if not _safe_key(key) or len(payload) > MAX_RECEIPT_BYTES:
        raise ValueError("Invalid curriculum control-object destination or size")
    try:
        prior = _bounded_get(client, bucket, key, MAX_RECEIPT_BYTES)
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404 and str(
            response.get("Error", {}).get("Code", "")
        ) not in {"404", "NoSuchKey", "NotFound"}:
            raise
    else:
        if prior != payload:
            raise ValueError("Existing curriculum handoff differs; refusing replacement")
        return payload
    try:
        client.put_object(
            Bucket=bucket, Key=key, Body=payload, ContentLength=len(payload), IfNoneMatch="*"
        )
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
            raise
    if _bounded_get(client, bucket, key, MAX_RECEIPT_BYTES) != payload:
        raise ValueError("Curriculum publication did not preserve the complete immutable receipt")
    return payload


def wait_for_curriculum(
    args, audio_client, checkpoint_client, *, sleep=time.sleep, monotonic=time.monotonic
):
    if (
        not math.isfinite(args.max_wait_seconds)
        or not 0 < args.max_wait_seconds <= 21600
        or not math.isfinite(args.poll_seconds)
        or not 1 <= args.poll_seconds <= 300
        or not _safe_key(args.curriculum_ready_key)
        or args.curriculum_ready_key.startswith(args.clean_prefix + "/")
    ):
        raise ValueError("Invalid bounded CPU wait or curriculum destination")
    deadline = monotonic() + args.max_wait_seconds

    def remaining():
        ensure_start_deadline(args.latest_start_utc)
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise TimeoutError("CPU curriculum gate wait expired without Nat readiness")
        return seconds

    with _metadata_deadline(min(180, remaining())):
        original, original_descriptor = fetch_clean_handoff(
            checkpoint_client,
            bucket=args.checkpoint_bucket,
            prefix=args.clean_prefix,
            run_id=args.clean_run_id,
            index_sha256=args.index_sha256,
        )
    while True:
        with _metadata_deadline(min(30, remaining())):
            try:
                audio_client.head_object(Bucket=args.nat_bucket, Key=args.nat_ready_key)
                break
            except Exception as error:
                response = getattr(error, "response", {})
                code = str(response.get("Error", {}).get("Code", ""))
                if response.get("ResponseMetadata", {}).get(
                    "HTTPStatusCode"
                ) != 404 and code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
        sleep(min(args.poll_seconds, remaining()))
    nat = fetch_ready_manifest(
        audio_client,
        bucket=args.nat_bucket,
        endpoint=args.nat_endpoint,
        ready_key=args.nat_ready_key,
        manifest_path=args.nat_manifest_path,
        inventory_path=args.nat_inventory,
        timeout_seconds=min(180, remaining()),
    )
    with _metadata_deadline(min(180, remaining())):
        handoff, descriptor = fetch_clean_handoff(
            checkpoint_client,
            bucket=args.checkpoint_bucket,
            prefix=args.clean_prefix,
            run_id=args.clean_run_id,
            index_sha256=args.index_sha256,
        )
        if handoff != original or descriptor != original_descriptor:
            raise ValueError("Clean writer advanced while the curriculum gate was waiting")
        value = {
            "schema_version": 1,
            "status": "complete",
            "ready_for_training": True,
            "source_prefix": args.clean_prefix,
            "source_run_id": args.clean_run_id,
            "index_sha256": args.index_sha256,
            "source_handoff": descriptor,
            "source_checkpoint_sha256": handoff["checkpoint"]["checkpoint"]["sha256"],
            "source_optimizer_step": handoff["checkpoint"]["checkpoint"]["step"],
            "nat": {
                "bucket": args.nat_bucket,
                "endpoint": args.nat_endpoint,
                "ready_key": args.nat_ready_key,
                "manifest": nat["manifest"],
                "inventory_sha256": nat["inventory_sha256"],
            },
        }
        payload = _publish_immutable(
            checkpoint_client, args.checkpoint_bucket, args.curriculum_ready_key, value
        )
    remaining()
    _save_immutable(args.output / "curriculum-ready.json", payload)
    if os.environ.get("GMN_RESULT_PATH"):
        _save_immutable(Path(os.environ["GMN_RESULT_PATH"]), payload)
    print(
        json.dumps(
            {"ready_for_training": True, "source_optimizer_step": value["source_optimizer_step"]}
        ),
        flush=True,
    )
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-prefix", required=True)
    parser.add_argument("--clean-run-id", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--checkpoint-bucket", required=True)
    parser.add_argument("--checkpoint-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--curriculum-ready-key", required=True)
    parser.add_argument("--nat-bucket", required=True)
    parser.add_argument("--nat-ready-key", required=True)
    parser.add_argument("--nat-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--nat-manifest-path", type=Path, required=True)
    parser.add_argument("--nat-inventory", type=Path, default=INVENTORY)
    parser.add_argument("--max-wait-seconds", type=float, default=21600)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--latest-start-utc", type=utc_timestamp, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ensure_start_deadline(args.latest_start_utc)
    args.nat_endpoint = _endpoint(args.nat_endpoint)
    args.checkpoint_endpoint = _endpoint(args.checkpoint_endpoint)
    wait_for_curriculum(
        args,
        metadata_client(args.nat_endpoint),
        checkpoint_metadata_client(args.checkpoint_endpoint),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CPU curriculum gate failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None

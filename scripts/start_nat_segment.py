"""Start a Nat production segment after a complete private-S3 asset handoff.

The CPU staging job must succeed before this command is scheduled on a GPU.
This bootstrap verifies metadata before importing the training stack. It records
the user's explicit waiver of a separate GPU preflight; asset integrity, actual
training validation and durable checkpoint publication remain mandatory.

Pass bootstrap options first, then ``--`` and the usual train_segment.py options.
Credentials use boto3's standard AWS environment; checkpoint credentials are
handled separately by train_segment.py and never appear in metadata or logs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INVENTORY = ROOT / "docs/validation/wavjepa-nat-assets-2026-10-23.json"
MAX_RECEIPT_BYTES = 1024**2
MAX_MANIFEST_BYTES = 256 * 1024**2
WAIVER_REASON = "User explicitly requested starting Nat training without a separate GPU preflight."
SHA256 = re.compile(r"[a-f0-9]{64}")


def utc_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
            raise ValueError
        return timestamp
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Latest start must be an ISO timestamp in UTC") from None


def ensure_start_deadline(latest_start, *, now=None):
    if latest_start is not None and (now or datetime.now(timezone.utc)) >= latest_start:
        raise TimeoutError("Nat production start deadline has passed; no training was launched")


def _safe_key(value):
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and ".." not in PurePosixPath(value).parts
        and "\\" not in value
        and "\x00" not in value
        and str(PurePosixPath(value)) == value
    )


def _endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("Nat endpoint must be an HTTPS origin without credentials")
    return value.rstrip("/")


@contextlib.contextmanager
def _metadata_deadline(seconds):
    """Bound SDK retries and streaming reads together on the POSIX job process."""
    if not math.isfinite(seconds) or not 0 < seconds <= 240:
        raise ValueError("Metadata timeout must be in (0, 240] seconds")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_signal, _frame):
        raise TimeoutError("Nat metadata download exceeded its total time bound")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, min(seconds, previous_timer[0]) if previous_timer[0] else seconds
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def metadata_client(endpoint):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=_endpoint(endpoint),
        region_name=os.environ.get("AWS_REGION", "auto"),
        config=Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=20,
            retries={"mode": "standard", "total_max_attempts": 2},
            s3={"addressing_style": "path"},
        ),
    )


def _bounded_get(client, bucket, key, maximum, *, expected_size=None):
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        size = response.get("ContentLength")
        if (
            type(size) is not int
            or not 0 < size <= maximum
            or (expected_size is not None and size != expected_size)
        ):
            raise ValueError("Nat metadata object has an invalid or unexpected size")
        chunks, received = [], 0
        while chunk := body.read(min(1024**2, maximum - received + 1)):
            received += len(chunk)
            if received > maximum or received > size:
                raise ValueError("Nat metadata object exceeds its declared size")
            chunks.append(chunk)
        if received != size:
            raise ValueError("Nat metadata object is truncated")
        return b"".join(chunks)
    finally:
        body.close()


def _validate_ready(receipt, *, bucket, endpoint, ready_key, inventory, inventory_sha256):
    if not isinstance(receipt, dict):
        raise ValueError("Nat staging receipt must be a JSON object")
    exact = {
        "schema_version": 1,
        "status": "complete",
        "ready_for_training": True,
        "scene_count": 70000,
        "noise_count": 20000,
        "completed_components": 71,
        "expected_components": 71,
        "bucket": bucket,
        "endpoint": endpoint,
        "inventory_sha256": inventory_sha256,
        "brir_repository": inventory["brir"]["repository"],
        "brir_revision": inventory["brir"]["revision"],
    }
    if any(
        type(receipt.get(key)) is not type(value) or receipt[key] != value
        for key, value in exact.items()
    ):
        raise ValueError("Nat staging is incomplete or differs from the packaged source inventory")
    manifest = receipt.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Nat staging receipt has no manifest descriptor")
    key, size, digest = (manifest.get(field) for field in ("key", "bytes", "sha256"))
    prefix = str(PurePosixPath(ready_key).parent) + "/"
    if (
        not _safe_key(key)
        or not key.startswith(prefix)
        or key == ready_key
        or type(size) is not int
        or not 0 < size <= MAX_MANIFEST_BYTES
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
    ):
        raise ValueError("Nat manifest descriptor is invalid or outside the staging prefix")
    return manifest


def _validate_manifest(value, receipt):
    if not isinstance(value, dict):
        raise ValueError("Nat manifest must be a JSON object")
    provenance = value.get("provenance", {})
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("sample_rate") != 16000
        or value.get("split") != "train"
        or value.get("bucket") != receipt["bucket"]
        or value.get("endpoint") != receipt["endpoint"]
        or not isinstance(value.get("scenes"), list)
        or len(value["scenes"]) != receipt["scene_count"]
        or not isinstance(value.get("noise_files"), list)
        or len(value["noise_files"]) != receipt["noise_count"]
        or not isinstance(value.get("assets"), dict)
        or not value["assets"]
        or not isinstance(provenance, dict)
        or provenance.get("coverage") != "full"
        or provenance.get("remote_verified") is not True
        or provenance.get("brir_repository") != receipt["brir_repository"]
        or provenance.get("brir_revision") != receipt["brir_revision"]
    ):
        raise ValueError("Nat manifest lacks full, remotely verified training-asset coverage")


def _save_immutable(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Nat metadata destination must not be a symbolic link")
    if path.exists():
        with path.open("rb") as handle:
            old_payload = handle.read(len(payload) + 1)
        if old_payload != payload:
            raise FileExistsError("Nat metadata destination already contains different bytes")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_ready_manifest(
    client,
    *,
    bucket,
    endpoint,
    ready_key,
    manifest_path,
    inventory_path=INVENTORY,
    timeout_seconds=180,
):
    """Freeze the CPU job's complete, source-bound manifest before importing Torch."""
    endpoint = _endpoint(endpoint)
    if re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None or not _safe_key(
        ready_key
    ):
        raise ValueError("Nat bucket or readiness key is invalid")
    with Path(inventory_path).open("rb") as handle:
        inventory_bytes = handle.read(MAX_RECEIPT_BYTES + 1)
    if len(inventory_bytes) > MAX_RECEIPT_BYTES:
        raise ValueError("Nat source inventory exceeds the metadata bound")
    inventory = json.loads(inventory_bytes)
    with _metadata_deadline(timeout_seconds):
        ready_bytes = _bounded_get(client, bucket, ready_key, MAX_RECEIPT_BYTES)
        receipt = json.loads(ready_bytes)
        descriptor = _validate_ready(
            receipt,
            bucket=bucket,
            endpoint=endpoint,
            ready_key=ready_key,
            inventory=inventory,
            inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        )
        payload = _bounded_get(
            client,
            bucket,
            descriptor["key"],
            MAX_MANIFEST_BYTES,
            expected_size=descriptor["bytes"],
        )
        if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ValueError("Nat manifest SHA256 differs from the completed staging receipt")
        _validate_manifest(json.loads(payload), receipt)
    manifest_path = Path(manifest_path)
    _save_immutable(manifest_path, payload)
    _save_immutable(manifest_path.with_name(manifest_path.name + ".ready.json"), ready_bytes)
    return receipt


def segment_command(segment_args, *, manifest_path, manifest_sha256, endpoint, curriculum=None):
    """Protect the selected Nat model, frozen inputs and explicit preflight waiver."""
    reserved = {
        "--nat",
        "--naturalistic-manifest",
        "--naturalistic-manifest-sha256",
        "--skip-streaming-acceptance",
        "--streaming-acceptance-waiver-reason",
        "--data-endpoint",
        "--streaming-smoke-report",
        "--worker-plan",
        "--execute-started",
        "--plan-only",
        "--curriculum-source-prefix",
        "--curriculum-source-run-id",
        "--curriculum-source-checkpoint-sha256",
    }
    if curriculum is not None:
        reserved.add("--global-batch-size")
    if any(argument.split("=", 1)[0] in reserved for argument in segment_args):
        raise ValueError("Segment arguments must not override bootstrap-owned Nat inputs")
    command = [
        sys.executable,
        str(ROOT / "scripts/train_segment.py"),
        "--nat",
        "--naturalistic-manifest",
        str(manifest_path),
        "--naturalistic-manifest-sha256",
        manifest_sha256,
        "--data-endpoint",
        endpoint,
        "--skip-streaming-acceptance",
        "--streaming-acceptance-waiver-reason",
        WAIVER_REASON,
        *segment_args,
    ]
    if curriculum is not None:
        global_batch = curriculum.get("source_global_batch", 16)
        if type(global_batch) is not int or global_batch < 4 or global_batch % 4:
            raise ValueError("Curriculum source global batch must be a positive multiple of four")
        command += [
            "--global-batch-size",
            str(global_batch),
            "--curriculum-source-prefix",
            curriculum["source_prefix"],
            "--curriculum-source-run-id",
            curriculum["source_run_id"],
            "--curriculum-source-checkpoint-sha256",
            curriculum["source_checkpoint_sha256"],
        ]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--nat-bucket", required=True)
    parser.add_argument("--nat-ready-key", required=True)
    parser.add_argument("--nat-manifest-path", type=Path, required=True)
    parser.add_argument("--nat-endpoint", default="https://storage.givemeanode.com")
    parser.add_argument("--nat-inventory", type=Path, default=INVENTORY)
    parser.add_argument("--metadata-timeout-seconds", type=float, default=180)
    parser.add_argument("--curriculum-ready-key")
    parser.add_argument("--curriculum-source-prefix")
    parser.add_argument("--curriculum-source-run-id")
    parser.add_argument(
        "--latest-start-utc",
        type=utc_timestamp,
        help="Refuse delayed starts at or after this UTC timestamp, before downloads and training",
    )
    parser.add_argument("segment_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    segment_args = args.segment_args[1:] if args.segment_args[:1] == ["--"] else args.segment_args
    if not segment_args:
        parser.error("Supply the bounded production segment arguments after --")
    curriculum_fields = [
        args.curriculum_ready_key,
        args.curriculum_source_prefix,
        args.curriculum_source_run_id,
    ]
    if any(curriculum_fields) and not all(curriculum_fields):
        parser.error(
            "Curriculum readiness key, source prefix and source run ID must be supplied together"
        )
    if args.curriculum_ready_key and any(
        argument.split("=", 1)[0] == "--global-batch-size" for argument in segment_args
    ):
        raise ValueError("Segment arguments must not override the verified curriculum global batch")
    ensure_start_deadline(args.latest_start_utc)
    endpoint = _endpoint(args.nat_endpoint)
    # Validate protected arguments before any network or file operation.
    segment_command(
        segment_args,
        manifest_path=args.nat_manifest_path,
        manifest_sha256="0" * 64,
        endpoint=endpoint,
    )
    curriculum = None
    with _metadata_deadline(args.metadata_timeout_seconds):
        receipt = fetch_ready_manifest(
            metadata_client(endpoint),
            bucket=args.nat_bucket,
            endpoint=endpoint,
            ready_key=args.nat_ready_key,
            manifest_path=args.nat_manifest_path,
            inventory_path=args.nat_inventory,
            timeout_seconds=args.metadata_timeout_seconds,
        )
        if args.curriculum_ready_key:
            from scripts.wait_nat_curriculum import (
                checkpoint_metadata_client,
                fetch_curriculum_handoff,
            )

            runner_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            runner_parser.add_argument("--checkpoint-bucket", required=True)
            runner_parser.add_argument("--checkpoint-prefix", required=True)
            runner_parser.add_argument(
                "--checkpoint-endpoint", default="https://storage.givemeanode.com"
            )
            runner_parser.add_argument("--index-sha256", required=True)
            runner, _ = runner_parser.parse_known_args(segment_args)
            if args.curriculum_ready_key.startswith(runner.checkpoint_prefix + "/"):
                raise ValueError(
                    "Curriculum control object must be outside the destination checkpoint prefix"
                )
            curriculum = fetch_curriculum_handoff(
                checkpoint_metadata_client(runner.checkpoint_endpoint),
                bucket=runner.checkpoint_bucket,
                ready_key=args.curriculum_ready_key,
                source_prefix=args.curriculum_source_prefix,
                source_run_id=args.curriculum_source_run_id,
                index_sha256=runner.index_sha256,
                nat_bucket=args.nat_bucket,
                nat_endpoint=endpoint,
                nat_ready_key=args.nat_ready_key,
                nat_receipt=receipt,
            )
    ensure_start_deadline(args.latest_start_utc)
    print(
        json.dumps(
            {
                "event": "nat_production_inputs_ready",
                "scene_count": receipt["scene_count"],
                "noise_count": receipt["noise_count"],
                "manifest_sha256": receipt["manifest"]["sha256"],
                "separate_gpu_preflight": "waived_by_user",
                "hardware_acceptance_passed": False,
                "curriculum_source_step": curriculum["source_optimizer_step"]
                if curriculum
                else None,
                "curriculum_global_batch": curriculum.get("source_global_batch", 16)
                if curriculum
                else None,
                "source_batch_transition": curriculum.get("source_batch_transition")
                if curriculum
                else None,
            }
        ),
        flush=True,
    )
    return subprocess.run(
        segment_command(
            segment_args,
            manifest_path=args.nat_manifest_path,
            manifest_sha256=receipt["manifest"]["sha256"],
            endpoint=endpoint,
            curriculum=curriculum,
        ),
        check=False,
    ).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # SDK exceptions can include endpoint or signed-request details.
        print(f"Nat production bootstrap failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None

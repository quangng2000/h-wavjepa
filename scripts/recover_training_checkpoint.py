"""Recover a pinned captured clean-training checkpoint using CPU only.

The caller supplies the private artifact URL through
HWAVJEPA_RECOVERY_ARTIFACT_URL and a separate checkpoint-writer credential pair.
No resources, bucket ownership, or curriculum handoff are created. The original
artifact remains the recovery fallback; only small receipts enter --output.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.checkpoint_transfer import safe_exception_details  # noqa: E402
from scripts.segment_checkpoint import (  # noqa: E402
    SHA,
    CheckpointRepository,
    checkpoint_client,
)
from scripts.train_segment import verify_resume_state  # noqa: E402
from src.stage_audioset import MAX_OBJECT_BYTES  # noqa: E402
from src.utils.runtime import atomic_json  # noqa: E402

MAX_ARTIFACT_BYTES = 5 * 1024**3
MAX_PLAN_BYTES = 1024**2
MAX_MEMBERS = 10000
CHECKPOINT_NAME = re.compile(r"step_([0-9]{7})\.ckpt")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _event(phase, **fields):
    print(json.dumps({"event": "checkpoint_recovery", "phase": phase, **fields}), flush=True)


def _artifact_url(value):
    parsed = urlparse(value or "")
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("Recovery artifact URL must be an HTTPS URL without user information")
    return value


def _copy_checked(source, destination, expected_bytes, check):
    digest, count = hashlib.sha256(), 0
    with destination.open("xb") as output:
        while True:
            check()
            chunk = source.read(1024**2)
            if not chunk:
                break
            count += len(chunk)
            if count > expected_bytes:
                raise ValueError("Recovery input exceeds its pinned byte count")
            output.write(chunk)
            digest.update(chunk)
        output.flush()
        os.fsync(output.fileno())
    if count != expected_bytes:
        raise ValueError("Recovery input does not match its pinned byte count")
    return digest.hexdigest()


def download_artifact(url, destination, *, expected_sha256, expected_bytes, check):
    """Fetch bounded bytes without exposing the signed URL in diagnostics."""
    url = _artifact_url(url)
    temporary = destination.with_suffix(".partial")
    try:
        check()
        request = Request(url, headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=60) as source:
            _artifact_url(source.geturl())
            if source.status != 200:
                raise ValueError("Artifact download did not return a complete object")
            length = source.headers.get("Content-Length")
            if length is not None and int(length) != expected_bytes:
                raise ValueError("Artifact response length does not match its pinned byte count")
            digest = _copy_checked(source, temporary, expected_bytes, check)
        if digest != expected_sha256:
            raise ValueError("Recovery artifact SHA256 mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _member_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ValueError("Unsafe recovery archive member path")
    return path


def extract_checkpoint(archive, work_dir, *, expected_step, check):
    """Inspect all headers, then write exactly two regular files without extract()."""
    with tarfile.open(archive, mode="r:*") as source:
        members, total_bytes = {}, 0
        for member in source:
            check()
            name = _member_path(member.name)
            if name in members or len(members) >= MAX_MEMBERS:
                raise ValueError("Duplicate or excessive recovery archive members")
            if not (member.isdir() or member.isfile() or member.issym()):
                raise ValueError("Recovery archive contains a special or hardlinked member")
            if member.issym():
                target = _member_path(member.linkname)
                if name.name != "last.ckpt" or len(target.parts) != 1:
                    raise ValueError("Recovery archive contains an unsupported symlink")
                if not CHECKPOINT_NAME.fullmatch(target.name):
                    raise ValueError("Recovery checkpoint symlink target has an invalid name")
            if member.isfile():
                if member.size < 0 or member.sparse is not None:
                    raise ValueError("Recovery archive contains invalid or sparse file metadata")
                total_bytes += member.size
                if total_bytes > MAX_ARTIFACT_BYTES:
                    raise ValueError("Recovery archive expands beyond the output-artifact bound")
            members[name] = member
        plans = [name for name in members if name.name == "segment-plan.json"]
        if len(plans) != 1:
            raise ValueError("Recovery artifact must contain exactly one segment plan")
        plan_name = plans[0]
        plan_member = members[plan_name]
        if not plan_member.isfile() or not 0 < plan_member.size <= MAX_PLAN_BYTES:
            raise ValueError("Recovery segment plan must be a bounded regular file")
        checkpoint_dir = plan_name.parent / "checkpoints"
        candidates = []
        for name, member in members.items():
            if name.parent != checkpoint_dir:
                continue
            match = CHECKPOINT_NAME.fullmatch(name.name)
            if match and member.isfile():
                step = int(match[1])
                if expected_step is None or step == expected_step:
                    candidates.append((name, step))
            elif name.suffix == ".ckpt" and name.name != "last.ckpt":
                raise ValueError("Recovery checkpoint record has an unexpected filename or type")
        if len(candidates) != 1:
            raise ValueError("Recovery artifact needs one unambiguous complete checkpoint")
        checkpoint_name, step = candidates[0]
        checkpoint_member = members[checkpoint_name]
        if not 1 <= step <= 375000 or not 0 < checkpoint_member.size <= MAX_OBJECT_BYTES:
            raise ValueError("Recovery checkpoint step or size is outside the supported bounds")
        plan_path = work_dir / "segment-plan.json"
        checkpoint_path = work_dir / checkpoint_name.name
        for member, path in ((plan_member, plan_path), (checkpoint_member, checkpoint_path)):
            check()
            with contextlib.closing(source.extractfile(member)) as handle:
                digest = _copy_checked(handle, path, member.size, check)
        return json.loads(plan_path.read_text()), checkpoint_path, step, digest


def repository_from_plan(plan, check):
    """Build the exact existing owner locally; no ownership adoption is allowed."""
    if not isinstance(plan, dict):
        raise ValueError("Recovery requires a plain segment plan")
    options = plan.get("repository_options")
    expected_options = {
        "bucket",
        "prefix",
        "run_id",
        "index_sha256",
        "base_data_seed",
        "global_batch",
        "naturalistic_manifest_sha256",
        "clean_binaural",
        "curriculum_source",
        "endpoint",
    }
    if not isinstance(options, dict) or set(options) != expected_options:
        raise ValueError("Recovery segment plan has unexpected repository settings")
    if (
        plan.get("clean_binaural") is not True
        or plan.get("nat") is not False
        or plan.get("temporal") is not True
        or plan.get("model_variant") != "clean_binaural_temporal_aggregation"
        or plan.get("global_batch") != 16
        or options.get("global_batch") != 16
        or options.get("clean_binaural") is not True
        or plan.get("naturalistic_manifest_sha256") is not None
        or options.get("naturalistic_manifest_sha256") is not None
        or plan.get("curriculum_source") is not None
        or options.get("curriculum_source") is not None
        or plan.get("index_sha256") != options.get("index_sha256")
        or plan.get("base_data_seed") != options.get("base_data_seed")
        or plan.get("stream_positions_checkpointed") is not False
        or plan.get("data_seed_policy") != "base_plus_resume_step"
        or type(plan.get("resume_step")) is not int
        or not 0 <= plan["resume_step"] < 375000
        or type(plan.get("resolved_data_seed")) is not int
        or plan["resolved_data_seed"] != options.get("base_data_seed", -1) + plan["resume_step"]
        or not isinstance(plan.get("segment_id"), str)
        or not SAFE_ID.fullmatch(plan["segment_id"])
    ):
        raise ValueError("Recovery plan does not match the owned clean training run")
    settings = dict(options)
    endpoint = settings.pop("endpoint")
    # Constructor validation is local. Credentials and network are used only
    # after complete model/training-state validation has succeeded.
    repository = CheckpointRepository(None, **settings, check=check, managed_upload=True)
    return repository, endpoint


def verify_full_model(path):
    """Require finite tensors and strict reconstruction, beyond counter checks."""
    from src.dynamics_inference import load_model

    model = load_model(path, device="cpu")
    del model
    gc.collect()


def run_recovery(args):
    if not SHA.fullmatch(args.artifact_sha256) or not 0 < args.artifact_bytes <= MAX_ARTIFACT_BYTES:
        raise ValueError("Pin the artifact SHA256 and a positive size of at most 5 GiB")
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 1770:
        raise ValueError("Recovery timeout must be positive and at most 1770 seconds")
    if args.checkpoint_step is not None and not 1 <= args.checkpoint_step <= 375000:
        raise ValueError("Expected checkpoint step must be between 1 and 375000")
    output, work = args.output.resolve(), args.work_dir.resolve()
    if output == work or work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError("Recovery output and temporary work directories must be separate")
    url = _artifact_url(os.environ.get("HWAVJEPA_RECOVERY_ARTIFACT_URL"))
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.timeout_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Checkpoint recovery exceeded its time limit")

    phase = "download"
    try:
        _event(phase, bytes=args.artifact_bytes)
        archive = work / "artifact.tar"
        download_artifact(
            url,
            archive,
            expected_sha256=args.artifact_sha256,
            expected_bytes=args.artifact_bytes,
            check=check,
        )
        phase = "extract"
        _event(phase)
        plan, checkpoint, step, digest = extract_checkpoint(
            archive, work, expected_step=args.checkpoint_step, check=check
        )
        archive.unlink()
        repository, endpoint = repository_from_plan(plan, check)
        if step <= plan["resume_step"]:
            raise ValueError("Recovered checkpoint does not advance its original segment")
        record = {
            "step": step,
            "sha256": digest,
            "bytes": checkpoint.stat().st_size,
            "key": f"{repository.prefix}/checkpoints/step_{step:07d}_{digest}.ckpt",
        }
        phase = "validate_training_state"
        _event(phase, step=step, bytes=record["bytes"])
        verify_resume_state(
            checkpoint,
            {"owner": repository.owner, "checkpoint": record},
            plan["index_sha256"],
            clean_binaural=True,
        )
        check()
        phase = "validate_full_model"
        _event(phase, step=step)
        verify_full_model(checkpoint)
        check()
        phase = "publish"
        _event(phase, step=step)
        repository.store.client = checkpoint_client(endpoint)
        repository.assert_owner()
        previous = repository.latest()
        if previous and previous["checkpoint"]["step"] >= step:
            if any(previous["checkpoint"].get(key) != value for key, value in record.items()):
                raise ValueError("Remote latest is newer or differs from the recovered checkpoint")
            # Idempotent recovery still performs a complete remote SHA256 readback.
            repository.store.verify(record["key"], digest, record["bytes"])
            receipt = previous
        else:
            receipt = repository.publish(
                checkpoint,
                step,
                segment_id=plan["segment_id"],
                data_seed=plan["resolved_data_seed"],
            )
        check()
        if repository.latest() != receipt:
            raise ValueError("Recovered receipt changed before final verification")
        result = {
            "recovered": True,
            "artifact_sha256": args.artifact_sha256,
            "artifact_bytes": args.artifact_bytes,
            "source_segment_id": plan["segment_id"],
            "full_training_state_verified": True,
            "model_state_strictly_loaded": True,
            "complete_remote_readback_verified": True,
            "stream_positions_checkpointed": False,
            "checkpoint": receipt,
        }
        atomic_json(output / "recovery-success.json", result)
        _event("completed", step=step, sha256=digest, bytes=record["bytes"])
        return result
    except Exception as error:
        failure = {
            "event": "checkpoint_recovery_failed",
            "phase": phase,
            **safe_exception_details(error),
        }
        atomic_json(output / "recovery-failed.json", failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        shutil.rmtree(work)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--artifact-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=1740)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 1770:
        parser.error("timeout-seconds must be positive and at most 1770")

    def hard_stop(signum, frame):
        # The process owns all transfer threads. Exit the process, rather than
        # waiting for blocked SDK threads beyond the 30-minute CPU-job budget.
        os.write(
            2,
            b'{"event":"checkpoint_recovery_failed","phase":"deadline","exception_chain":["TimeoutError"]}\n',
        )
        os._exit(124)

    signal.signal(signal.SIGALRM, hard_stop)
    signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds + 15)
    try:
        result = run_recovery(args)
        if os.environ.get("GMN_RESULT_PATH"):
            record = result["checkpoint"]["checkpoint"]
            atomic_json(
                Path(os.environ["GMN_RESULT_PATH"]),
                {
                    "recovered": True,
                    "checkpoint_step": record["step"],
                    "checkpoint_sha256": record["sha256"],
                    "checkpoint_bytes": record["bytes"],
                    "full_training_state_verified": True,
                    "complete_remote_readback_verified": True,
                },
            )
    except Exception as error:
        # Preflight failures occur before an owned output directory exists.
        failure = {"recovered": False, **safe_exception_details(error)}
        if os.environ.get("GMN_RESULT_PATH"):
            # Preserve the original safe diagnostic if a platform result path is
            # itself unavailable, without exposing filesystem/SDK error text.
            with contextlib.suppress(Exception):
                atomic_json(Path(os.environ["GMN_RESULT_PATH"]), failure)
        print(json.dumps({"event": "checkpoint_recovery_exit", **failure}), flush=True)
        raise SystemExit(1) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()

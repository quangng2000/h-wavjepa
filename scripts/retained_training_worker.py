"""Keep one bounded four-GPU task-set allocation through explicitly requested repairs.

The immutable, SHA-pinned plan supplies the only launch command. An open SFCompute
task set supplies the initial ``train`` task and later explicit ``repair`` tasks.
This worker never rents compute, retries training automatically, extends its
allocation deadline, or changes checkpoint ownership. Seal the task set externally
after this worker exits; exit zero alone does not mean its pool was sealed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_PATCH_BYTES = 256 * 1024
MAX_TASK_BYTES = 1024 * 1024
NCCL_ENVIRONMENT = {"NCCL_NVLS_ENABLE", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


class TaskAPIError(RuntimeError):
    """Never carry response bodies, signed URLs, credentials or SDK messages."""

    def __init__(self, operation, status=None):
        self.status = status
        super().__init__(f"Task metadata operation {operation} failed (status={status})")


class ProcessCleanupError(RuntimeError):
    """A surviving rank means no further checkpoint writer may be started."""


class TaskAPI:
    def __init__(self, metadata_url, remaining, *, clock=time.monotonic, sleep=time.sleep):
        self.metadata_url = metadata_url.rstrip("/")
        self.remaining = remaining
        self.clock = clock
        self.sleep = sleep

    def request(self, url, *, method="POST", value=None, metadata=True):
        retry_deadline = self.clock() + min(30.0, self.remaining())
        for attempt in range(3):
            timeout = min(10.0, retry_deadline - self.clock(), self.remaining())
            if timeout <= 0:
                raise TaskAPIError(method)
            try:
                return self.request_once(
                    url, method=method, value=value, metadata=metadata, timeout=timeout
                )
            except TaskAPIError as error:
                if error.status not in {None, 408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise
                delay = min(2**attempt, retry_deadline - self.clock(), self.remaining())
                if delay <= 0:
                    raise
                self.sleep(delay)
        raise AssertionError("Unreachable task request state")

    def request_once(self, url, *, method, value, metadata, timeout):
        body = None if value is None else json.dumps(value).encode()
        headers = {"Content-Type": "application/json"}
        if metadata:
            headers["Metadata-Flavor"] = "givemeanode"
        request = urllib.request.Request(url, method=method, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                data = response.read(MAX_TASK_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise TaskAPIError(method, error.code) from None
        except (OSError, ValueError):
            raise TaskAPIError(method) from None
        if not 200 <= status < 300 or len(data) > MAX_TASK_BYTES:
            raise TaskAPIError(method, status)
        if not metadata:
            return status, {}
        try:
            return status, json.loads(data or "{}")
        except (ValueError, UnicodeError):
            raise TaskAPIError(method, status) from None

    def post(self, path, value=None):
        return self.request(self.metadata_url + path, value=value)

    def lease(self):
        status, result = self.post("/v1/tasks/lease", {"count": 1})
        if status == 204:
            return None
        tasks = result.get("tasks")
        if not isinstance(tasks, list) or len(tasks) > 1:
            raise TaskAPIError("lease")
        return tasks

    def extend(self, task):
        # The public protocol specifies this POST, without a request body. Its
        # success is checked immediately before starting and regularly thereafter.
        self.post(f"/v1/tasks/{task['id']}/extend")

    def complete(self, task, receipt):
        self.request(task["output_put_url"], method="PUT", value=receipt, metadata=False)
        self.post(f"/v1/tasks/{task['id']}/done")


def load_plan(path, expected_sha256):
    raw = Path(path).read_bytes()
    if not SHA256.fullmatch(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Immutable launch plan SHA256 mismatch")
    plan = json.loads(raw)
    required = {
        "schema_version",
        "argv",
        "output_root",
        "work_root",
        "segment_prefix",
        "max_runtime_seconds",
        "repair_wait_seconds",
        "max_training_attempts",
        "budget_usd",
        "maximum_rate_usd_per_hour",
        "allowed_patch_paths",
    }
    if set(plan) != required or plan["schema_version"] != 1:
        raise ValueError("Unexpected immutable plan schema")
    argv = plan["argv"]
    if isinstance(argv, list) and argv[:2] == ["python", "-u"]:
        argv = argv[:1] + argv[2:]
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or argv[:2] != ["python", "scripts/train_segment.py"]
        or any(not isinstance(item, str) or "\0" in item for item in argv)
        or "--resume" not in argv
        or "--exclusive-checkpoint-writer" not in argv
        or "--acknowledge-stream-restart" not in argv
        or any(item in argv for item in ("--worker-plan", "--execute-started", "--plan-only"))
    ):
        raise ValueError("Plan must explicitly resume the fixed training entry point")
    for name in (
        "max_runtime_seconds",
        "repair_wait_seconds",
        "budget_usd",
        "maximum_rate_usd_per_hour",
    ):
        value = plan[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid allocation bound")
    if (
        plan["max_runtime_seconds"] > 7 * 3600
        or plan["repair_wait_seconds"] > 1800
        or type(plan["max_training_attempts"]) is not int
        or not 1 <= plan["max_training_attempts"] <= 3
        or not IDENTIFIER.fullmatch(plan["segment_prefix"])
        or (math.ceil(plan["max_runtime_seconds"] / 60) + 5)
        / 60
        * plan["maximum_rate_usd_per_hour"]
        > plan["budget_usd"] + 1e-9
    ):
        raise ValueError("Plan exceeds retention, attempt or spending bounds")
    output, work = Path(plan["output_root"]), Path(plan["work_root"])
    if not output.is_absolute() or not work.is_absolute() or output == work:
        raise ValueError("Output and scratch roots must be distinct absolute paths")
    if output in work.parents or work in output.parents:
        raise ValueError("Scratch cache must be outside captured output")
    paths = plan["allowed_patch_paths"]
    if not isinstance(paths, list) or len(paths) != len(set(paths)):
        raise ValueError("Patch allowlist must contain unique paths")
    for path in paths:
        safe_patch_path(path)
    return plan


def safe_patch_path(path):
    if not isinstance(path, str):
        raise ValueError("Invalid patch path")
    relative = Path(path)
    target = ROOT / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in {"scripts", "src", "configs"}
        or relative.suffix not in {".py", ".yaml", ".yml"}
        or target.resolve() != target
        or target == Path(__file__).resolve()
        or not target.is_file()
    ):
        raise ValueError("Patch path must be an existing, non-symlink allowed source file")
    return target


def apply_task(task, plan, plan_sha256, attempts):
    if not isinstance(task, dict) or not IDENTIFIER.fullmatch(str(task.get("id", ""))):
        raise ValueError("Invalid task identifier")
    if not isinstance(task.get("output_put_url"), str) or not task["output_put_url"].startswith(
        "https://"
    ):
        raise ValueError("Task requires its provider-issued HTTPS result upload")
    params = task.get("params")
    if isinstance(params, str) and len(params.encode()) <= MAX_TASK_BYTES:
        params = json.loads(params)
    if not isinstance(params, dict) or len(json.dumps(params).encode()) > MAX_TASK_BYTES:
        raise ValueError("Invalid task parameters")
    if set(params) - {"action", "plan_sha256", "patches", "environment", "after_attempt"}:
        raise ValueError("Task cannot override launch arguments or allocation limits")
    if params.get("plan_sha256") != plan_sha256:
        raise ValueError("Task targets a different immutable plan")
    if attempts == 0:
        if params.get("action") != "train" or set(params) != {"action", "plan_sha256"}:
            raise ValueError("First task must be the unchanged initial training task")
    elif params.get("action") != "repair" or params.get("after_attempt") != attempts:
        raise ValueError("A repair must explicitly target the preceding failed attempt")
    environment = params.get("environment", {})
    if not isinstance(environment, dict) or any(
        name not in NCCL_ENVIRONMENT or value not in {"0", "1"}
        for name, value in environment.items()
    ):
        raise ValueError("Only whitelisted NCCL boolean environment repairs are allowed")
    patches = params.get("patches", [])
    if not isinstance(patches, list):
        raise ValueError("Invalid patch list")
    verified = []
    seen = set()
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != {
            "path",
            "previous_sha256",
            "sha256",
            "content_base64",
        }:
            raise ValueError("Patch must include exact old and new SHA256 values")
        path = patch["path"]
        if path not in plan["allowed_patch_paths"] or path in seen:
            raise ValueError("Patch path absent from immutable allowlist or repeated")
        seen.add(path)
        target = safe_patch_path(path)
        if (
            not SHA256.fullmatch(str(patch["previous_sha256"]))
            or hashlib.sha256(target.read_bytes()).hexdigest() != patch["previous_sha256"]
        ):
            raise ValueError("Patch predecessor SHA256 mismatch")
        content = base64.b64decode(patch["content_base64"], validate=True)
        if len(content) > MAX_PATCH_BYTES or hashlib.sha256(content).hexdigest() != patch["sha256"]:
            raise ValueError("Patch content size or SHA256 mismatch")
        if target.suffix == ".py":
            compile(content, path, "exec")
        verified.append((target, content))
    # Validate the entire repair before mutating any source file.
    for target, content in verified:
        temporary = target.with_suffix(target.suffix + ".repair-tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return environment


def replace_option(argv, option, value):
    if argv.count(option) != 1:
        raise ValueError("Pinned command needs exactly one instance of each bounded option")
    index = argv.index(option)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError("Pinned command option lacks a value")
    argv[index + 1] = str(value)


def attempt_command(plan, attempt, remaining, elapsed):
    # One minute inside the absolute worker limit is reserved for tree cleanup.
    seconds = remaining - 60
    if seconds <= 1920:
        raise TimeoutError("Insufficient remaining allocation for safe finalization")
    argv = list(plan["argv"])
    argv[0] = sys.executable
    output = Path(plan["output_root"]) / f"attempt-{attempt:02d}"
    work = Path(plan["work_root"]) / f"attempt-{attempt:02d}"
    for option, value in {
        "--segment-hours": seconds / 3600,
        "--budget-usd": plan["budget_usd"] - elapsed / 3600 * plan["maximum_rate_usd_per_hour"],
        "--maximum-rate-usd-per-hour": plan["maximum_rate_usd_per_hour"],
        "--segment-id": f"{plan['segment_prefix']}-{attempt:02d}",
        "--output": output,
        "--work-dir": work,
    }.items():
        replace_option(argv, option, value)
    if output.exists() or work.exists():
        raise FileExistsError("A retained attempt must use fresh output and scratch directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)
    return argv, output


def run_child(argv, *, environment, deadline, renew, clock=time.monotonic):
    import psutil

    from scripts.train_segment import remember_descendants, stop_process_tree

    process = subprocess.Popen(argv, cwd=ROOT, env=environment, start_new_session=True)
    root = psutil.Process(process.pid)
    owned = {(root.pid, root.create_time()): root}
    next_renewal = clock() + 60
    try:
        while True:
            remember_descendants(owned)
            code = process.poll()
            if code is not None:
                return code
            if clock() >= deadline:
                raise TimeoutError("Absolute retained-allocation deadline reached")
            if clock() >= next_renewal:
                renew()
                next_renewal = clock() + 60
            try:
                process.wait(timeout=min(1, max(0.01, deadline - clock())))
            except subprocess.TimeoutExpired:
                pass
    finally:
        # No repair task can start until every surviving rank/data worker dies.
        try:
            stop_process_tree(process, owned)
        except Exception:
            raise ProcessCleanupError("Owned rank cleanup could not be verified") from None


def successful_result(path):
    result = json.loads(Path(path).read_text())
    step = result.get("final_optimizer_step", 0)
    if (
        result.get("segment_finished") is not True
        or step <= result.get("initial_optimizer_step", 0)
        or result.get("ema_updates") != step
        or result.get("checkpoint", {}).get("checkpoint", {}).get("step") != step
    ):
        raise ValueError("Training did not produce a matching verified checkpoint result")
    return result


def run_worker(plan, plan_sha256, api, *, clock=time.monotonic, sleep=time.sleep, launch=run_child):
    started = clock()
    deadline = started + plan["max_runtime_seconds"]
    attempts = 0
    waited = 0.0
    wait_started = clock()
    environment_repairs = {}
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    emit(
        "retained_worker_ready", plan_sha256=plan_sha256, max_attempts=plan["max_training_attempts"]
    )
    while clock() < deadline - 60 and attempts < plan["max_training_attempts"]:
        if waited + clock() - wait_started >= plan["repair_wait_seconds"]:
            break
        try:
            tasks = api.lease()
        except TaskAPIError as error:
            emit("task_metadata_unavailable", status=error.status)
            sleep(min(5, max(0, deadline - 60 - clock())))
            continue
        if tasks is None:
            break
        if not tasks:
            sleep(min(5, max(0, deadline - 60 - clock())))
            continue
        task = tasks[0]
        waited += clock() - wait_started
        receipt = {"training_success": False, "attempt": attempts, "plan_sha256": plan_sha256}
        result = None
        fatal_cleanup = False
        try:
            # Validate a provider lease before editing source or starting ranks.
            if not isinstance(task, dict) or not IDENTIFIER.fullmatch(str(task.get("id", ""))):
                raise ValueError("Malformed provider task")
            api.extend(task)
            environment_repairs.update(apply_task(task, plan, plan_sha256, attempts))
            # Keep old failed checkpoints on this allocated disk, without letting
            # multiple 2+ GiB snapshots overflow the platform's 5 GiB artifact cap.
            # Small task receipts remain captured. Nothing unverified is deleted.
            if attempts:
                prior = output_root / f"attempt-{attempts:02d}"
                archive = Path(plan["work_root"]) / "previous-attempt-output" / prior.name
                if prior.exists():
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(prior), str(archive))
            attempts += 1
            receipt["attempt"] = attempts
            argv, output = attempt_command(plan, attempts, deadline - clock(), clock() - started)
            child_result = output / "child-result.json"
            environment = (
                dict(os.environ) | environment_repairs | {"GMN_RESULT_PATH": str(child_result)}
            )
            emit(
                "retained_attempt_started",
                attempt=attempts,
                remaining_seconds=int(deadline - clock()),
            )
            code = launch(
                argv,
                environment=environment,
                deadline=deadline - 30,
                renew=lambda: api.extend(task),
                clock=clock,
            )
            receipt["exit_code"] = code
            if code == 0:
                result = successful_result(child_result)
                receipt.update(
                    training_success=True, final_optimizer_step=result["final_optimizer_step"]
                )
        except Exception as error:
            # Exception text may contain signed URLs or credential-bearing paths.
            receipt["error_type"] = type(error).__name__
            fatal_cleanup = isinstance(error, ProcessCleanupError)
        receipt["elapsed_seconds"] = clock() - started
        atomic_json(output_root / f"task-receipt-{attempts:02d}.json", receipt)
        receipt_accepted = False
        try:
            api.complete(task, receipt)
            receipt_accepted = True
        except Exception as error:
            emit(
                "task_receipt_publication_failed",
                status=getattr(error, "status", None),
                attempt=attempts,
            )
            # A successful clean child has sealed its handoff. Preserve that
            # evidence and never try to train its now-sealed prefix again.
        if fatal_cleanup:
            break
        if result is not None:
            result["retained_allocation"] = {
                "attempts": attempts,
                "elapsed_seconds": clock() - started,
                "repair_wait_seconds": waited,
                "plan_sha256": plan_sha256,
                "task_set_requires_external_seal": True,
                "task_receipt_accepted": receipt_accepted,
            }
            atomic_json(output_root / "retained-training-success.json", result)
            if not receipt_accepted:
                failure = {
                    "training_success": True,
                    "retained_reporting_failed": True,
                    "ready_for_nat": False,
                    "final_optimizer_step": result["final_optimizer_step"],
                    "verified_result_path": str(output_root / "retained-training-success.json"),
                }
                atomic_json(output_root / "retained-reporting-failure.json", failure)
                if os.environ.get("GMN_RESULT_PATH"):
                    atomic_json(Path(os.environ["GMN_RESULT_PATH"]), failure)
                emit(
                    "retained_training_reporting_failed",
                    final_optimizer_step=result["final_optimizer_step"],
                )
                return 1
            if os.environ.get("GMN_RESULT_PATH"):
                atomic_json(Path(os.environ["GMN_RESULT_PATH"]), result)
            emit(
                "retained_training_succeeded",
                attempt=attempts,
                final_optimizer_step=result["final_optimizer_step"],
            )
            return 0
        emit(
            "retained_waiting_for_explicit_repair",
            attempt=attempts,
            repair_wait_remaining_seconds=max(0, int(plan["repair_wait_seconds"] - waited)),
        )
        wait_started = clock()
    result = {
        "training_success": False,
        "retained_allocation_ended": True,
        "attempts": attempts,
        "elapsed_seconds": clock() - started,
    }
    atomic_json(output_root / "retained-training-stopped.json", result)
    if os.environ.get("GMN_RESULT_PATH"):
        atomic_json(Path(os.environ["GMN_RESULT_PATH"]), result)
    emit("retained_allocation_ended", attempts=attempts)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan, args.plan_sha256)
    started = time.monotonic()
    api = TaskAPI(
        os.environ["GMN_METADATA_URL"],
        lambda: started + plan["max_runtime_seconds"] - time.monotonic(),
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    return run_worker(plan, args.plan_sha256, api)


if __name__ == "__main__":
    raise SystemExit(main())

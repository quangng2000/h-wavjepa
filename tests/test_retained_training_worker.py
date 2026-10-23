"""CPU-only checks for bounded explicit repairs within one task-set allocation."""

import base64
import hashlib
import io
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import psutil
import pytest

from scripts import retained_training_worker as worker


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def plan(tmp_path):
    return {
        "schema_version": 1,
        "argv": [
            "python",
            "-u",
            "scripts/train_segment.py",
            "--resume",
            "--exclusive-checkpoint-writer",
            "--acknowledge-stream-restart",
            "--segment-hours",
            "1",
            "--budget-usd",
            "20",
            "--maximum-rate-usd-per-hour",
            "15",
            "--segment-id",
            "original",
            "--output",
            "/output/original",
            "--work-dir",
            "/workspace/original",
        ],
        "output_root": str(tmp_path / "output"),
        "work_root": str(tmp_path / "scratch"),
        "segment_prefix": "clean-retained",
        "max_runtime_seconds": 3600,
        "repair_wait_seconds": 1800,
        "max_training_attempts": 3,
        "budget_usd": 20,
        "maximum_rate_usd_per_hour": 15,
        "allowed_patch_paths": [],
    }


def pinned_plan(tmp_path, plan):
    raw = json.dumps(plan).encode()
    path = tmp_path / "plan.json"
    path.write_bytes(raw)
    return worker.load_plan(path, digest(raw)), digest(raw)


def task(action="train", *, number=1, plan_hash="a" * 64, **params):
    return {
        "id": f"task-{number}",
        "params": json.dumps({"action": action, "plan_sha256": plan_hash, **params}),
        "output_put_url": "https://storage.example/signed-result?secret=never-print",
    }


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class API:
    def __init__(self, tasks):
        self.tasks = iter(tasks)
        self.completed = []
        self.extended = []

    def lease(self):
        return next(self.tasks, [])

    def extend(self, task):
        self.extended.append(task["id"])

    def complete(self, task, result):
        self.completed.append((task["id"], result))


def success(environment, initial=250, final=500):
    value = {
        "segment_finished": True,
        "initial_optimizer_step": initial,
        "final_optimizer_step": final,
        "ema_updates": final,
        "checkpoint": {"checkpoint": {"step": final}},
        "ready_for_nat": True,
    }
    path = Path(environment["GMN_RESULT_PATH"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_pinned_plan_and_runtime_command_shrink_after_wait(tmp_path, plan):
    loaded, _ = pinned_plan(tmp_path, plan)
    assert loaded == plan
    first, _ = worker.attempt_command(plan, 1, 3600, 0)
    second, _ = worker.attempt_command(plan, 2, 2600, 1000)
    assert first[0] == sys.executable
    assert float(second[second.index("--segment-hours") + 1]) < float(
        first[first.index("--segment-hours") + 1]
    )
    assert float(second[second.index("--budget-usd") + 1]) < 20
    assert second[second.index("--output") + 1] != first[first.index("--output") + 1]
    # The segment runner owns creation with exist_ok=False; the wrapper must not
    # pre-create either per-attempt directory.
    assert not Path(first[first.index("--output") + 1]).exists()
    assert not Path(first[first.index("--work-dir") + 1]).exists()
    with pytest.raises(ValueError, match="SHA256"):
        worker.load_plan(tmp_path / "plan.json", "0" * 64)


@pytest.mark.parametrize(
    "change",
    [
        {"max_runtime_seconds": 25201},
        {"repair_wait_seconds": 1801},
        {"max_training_attempts": 4},
        {"budget_usd": 1},
        {"argv": ["sh", "-c", "anything"]},
    ],
)
def test_plan_cannot_expand_bounds_or_run_shell(tmp_path, plan, change):
    plan.update(change)
    with pytest.raises(ValueError):
        pinned_plan(tmp_path, plan)


def test_failure_waits_for_explicit_repair_then_preserves_success(plan, monkeypatch, tmp_path):
    clock = Clock()
    api = API([[task()], [], [], [task("repair", number=2, after_attempt=1)]])
    commands = []
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("GMN_RESULT_PATH", str(result_path))

    def launch(argv, **kwargs):
        commands.append(argv)
        clock.sleep(100)
        if len(commands) == 1:
            Path(kwargs["environment"]["GMN_RESULT_PATH"]).parent.mkdir(parents=True)
            return 1
        success(kwargs["environment"])
        return 0

    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 0
    )
    assert len(commands) == 2
    assert api.completed[0][1]["training_success"] is False
    assert api.completed[1][1]["training_success"] is True
    result = json.loads(result_path.read_text())
    assert result["ready_for_nat"] is True
    assert result["retained_allocation"]["repair_wait_seconds"] == 10
    assert result["retained_allocation"]["task_set_requires_external_seal"] is True
    assert (Path(plan["work_root"]) / "previous-attempt-output/attempt-01").is_dir()
    assert not (Path(plan["output_root"]) / "attempt-01").exists()


def test_no_automatic_retry_and_cumulative_wait_cap(plan):
    clock = Clock()
    api = API([[task()]])
    calls = []

    def launch(*args, **kwargs):
        calls.append(1)
        return 1

    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 1
    )
    assert len(calls) == 1
    assert clock.now == 1800


def test_stale_repair_cannot_restart_or_extend_deadline(plan):
    clock = Clock()
    api = API([[task()], [task("repair", number=2, after_attempt=0)], None])
    calls = []

    def launch(*args, **kwargs):
        calls.append(1)
        return 1

    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 1
    )
    assert len(calls) == 1
    assert api.completed[-1][1]["error_type"] == "ValueError"


def test_unverified_child_success_is_not_job_success(plan):
    clock = Clock()
    api = API([[task()], None])
    assert (
        worker.run_worker(
            plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=lambda *a, **k: 0
        )
        == 1
    )
    assert api.completed[0][1]["training_success"] is False


def test_cleanup_failure_never_starts_next_writer(plan):
    clock = Clock()
    api = API([[task()], [task("repair", number=2, after_attempt=1)]])

    def launch(*args, **kwargs):
        raise worker.ProcessCleanupError("still alive")

    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 1
    )
    assert len(api.completed) == 1


def test_sha_pinned_repairs_validate_all_files_before_mutating(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    directory = tmp_path / "scripts"
    directory.mkdir()
    target = directory / "model.py"
    target.write_bytes(b"value = 1\n")
    content = b"value = 2\n"
    plan["allowed_patch_paths"] = ["scripts/model.py"]
    patch = {
        "path": "scripts/model.py",
        "previous_sha256": digest(target.read_bytes()),
        "sha256": digest(content),
        "content_base64": base64.b64encode(content).decode(),
    }
    invalid = deepcopy(patch)
    invalid["path"] = "../outside.py"
    with pytest.raises(ValueError):
        worker.apply_task(
            task("repair", after_attempt=1, patches=[patch, invalid]), plan, "a" * 64, 1
        )
    assert target.read_bytes() == b"value = 1\n"
    env = worker.apply_task(
        task("repair", after_attempt=1, patches=[patch], environment={"NCCL_NVLS_ENABLE": "0"}),
        plan,
        "a" * 64,
        1,
    )
    assert target.read_bytes() == content
    assert env == {"NCCL_NVLS_ENABLE": "0"}
    with pytest.raises(ValueError, match="predecessor"):
        worker.apply_task(task("repair", after_attempt=1, patches=[patch]), plan, "a" * 64, 1)


def test_credentials_and_arbitrary_commands_rejected(plan):
    for params in ({"environment": {"AWS_SECRET_ACCESS_KEY": "x"}}, {"command": "sh"}):
        with pytest.raises(ValueError):
            worker.apply_task(task("repair", after_attempt=1, **params), plan, "a" * 64, 1)


def test_task_api_uploads_before_done_and_extend_has_no_body(monkeypatch):
    requests = []

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        requests.append(request)
        return Response(b"{}")

    monkeypatch.setattr(worker.urllib.request, "urlopen", urlopen)
    clock = Clock()
    api = worker.TaskAPI("http://metadata", lambda: 60, clock=clock, sleep=clock.sleep)
    value = task()
    api.extend(value)
    api.complete(value, {"training_success": False})
    assert requests[0].full_url.endswith("/extend")
    assert requests[0].data is None
    assert requests[0].headers["Metadata-flavor"] == "givemeanode"
    assert requests[1].method == "PUT"
    assert "Metadata-flavor" not in requests[1].headers
    assert requests[2].full_url.endswith("/done")


def test_http_errors_do_not_expose_signed_urls(monkeypatch):
    def urlopen(request, **kwargs):
        raise worker.urllib.error.HTTPError(
            "https://secret-url?secret=yes", 503, "secret body", {}, None
        )

    monkeypatch.setattr(worker.urllib.request, "urlopen", urlopen)
    api = worker.TaskAPI("http://metadata", lambda: 60)
    with pytest.raises(worker.TaskAPIError) as error:
        api.lease()
    assert error.value.status == 503
    assert "secret" not in str(error.value)


def test_real_cpu_child_tree_is_stopped_at_absolute_deadline(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
        f"open({str(pid_file)!r},'w').write(str(p.pid)); "
        "time.sleep(30)"
    )
    with pytest.raises(TimeoutError):
        worker.run_child(
            [sys.executable, "-c", code],
            environment=dict(os.environ),
            deadline=time.monotonic() + 2,
            renew=lambda: None,
        )
    if pid_file.exists():
        pid = int(pid_file.read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_transient_metadata_failures_retry_within_same_lease(monkeypatch):
    calls = []
    clock = Clock()

    class Response(io.BytesIO):
        status = 200

    def urlopen(request, **kwargs):
        calls.append(request)
        if len(calls) <= 2:
            raise worker.urllib.error.HTTPError("https://secret", 503, "secret", {}, None)
        return Response(b"{}")

    monkeypatch.setattr(worker.urllib.request, "urlopen", urlopen)
    api = worker.TaskAPI("http://metadata", lambda: 60, clock=clock, sleep=clock.sleep)
    api.extend(task())
    assert len(calls) == 3
    assert clock.now == 3
    assert all(request.full_url.endswith("/extend") for request in calls)


def test_lost_lease_is_never_retried(monkeypatch):
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        raise worker.urllib.error.HTTPError("https://secret", 409, "lost", {}, None)

    monkeypatch.setattr(worker.urllib.request, "urlopen", urlopen)
    api = worker.TaskAPI("http://metadata", lambda: 60)
    with pytest.raises(worker.TaskAPIError) as error:
        api.extend(task())
    assert error.value.status == 409
    assert len(calls) == 1


def test_three_attempt_ceiling_stops_even_with_more_repair_tasks(plan):
    clock = Clock()
    api = API([[task()]] + [[task("repair", number=n, after_attempt=n - 1)] for n in range(2, 5)])
    calls = []

    def launch(*args, **kwargs):
        calls.append(1)
        return 1

    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 1
    )
    assert len(calls) == 3


def test_completed_sealed_training_never_restarts_when_task_reporting_fails(
    plan, monkeypatch, tmp_path
):
    clock = Clock()
    api = API([[task()], [task("repair", number=2, after_attempt=1)]])
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("GMN_RESULT_PATH", str(result_path))
    calls = []

    def launch(*args, **kwargs):
        calls.append(1)
        success(kwargs["environment"])
        return 0

    def failed_report(*args):
        raise worker.TaskAPIError("done", 409)

    api.complete = failed_report
    assert (
        worker.run_worker(plan, "a" * 64, api, clock=clock, sleep=clock.sleep, launch=launch) == 1
    )
    assert len(calls) == 1
    result = json.loads(result_path.read_text())
    assert result["retained_reporting_failed"] is True
    assert result["ready_for_nat"] is False
    preserved = json.loads(
        (Path(plan["output_root"]) / "retained-training-success.json").read_text()
    )
    assert preserved["final_optimizer_step"] == 500
    assert preserved["retained_allocation"]["task_receipt_accepted"] is False

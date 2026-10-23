"""Exercise the actual shell argument and reject oversized or corrupted payloads."""

import hashlib
import random
import string
import subprocess
import sys

import pytest

from scripts.package_runtime_bootstrap import MAX_COMMAND_BYTES, package_bootstrap


def test_large_overlay_executes_through_one_shell_argument(tmp_path):
    source = b"# An embedded runtime overlay\n" * 10_000
    source += b"from pathlib import Path\nPath('started').write_text('verified')\n"
    assert len(source) > 128 * 1024
    command = package_bootstrap(source, working_directory=tmp_path, python=sys.executable)
    assert len(command.encode()) < MAX_COMMAND_BYTES
    subprocess.run(["/bin/sh", "-c", command], check=True, timeout=10)
    assert (tmp_path / "started").read_text() == "verified"


def test_incompressible_overlay_rejected_before_submission():
    source = ("# " + "".join(random.Random(42).choices(string.ascii_letters, k=150_000))).encode()
    with pytest.raises(ValueError, match="exceeds 64 KiB"):
        package_bootstrap(source)


def test_changed_digest_fails_before_overlay_execution(tmp_path):
    source = b"from pathlib import Path\nPath('started').touch()\n"
    command = package_bootstrap(source, working_directory=tmp_path, python=sys.executable)
    command = command.replace(hashlib.sha256(source).hexdigest(), "0" * 64)
    result = subprocess.run(["/bin/sh", "-c", command], capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"SHA256 mismatch" in result.stderr
    assert not (tmp_path / "started").exists()


def test_invalid_python_rejected_before_submission():
    with pytest.raises(SyntaxError):
        package_bootstrap(b"def invalid(:")

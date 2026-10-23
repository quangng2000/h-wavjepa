"""Package a Python runtime overlay below the shell's single-argument limit."""

import argparse
import base64
import hashlib
import shlex
import zlib
from pathlib import Path

# Linux commonly limits each exec argument to 128 KiB, including its NUL.
# Batch commands become one /bin/sh -c argument; leave substantial headroom.
MAX_COMMAND_BYTES = 64 * 1024


def package_bootstrap(source: bytes, *, working_directory="/app", python="python") -> str:
    compile(source, "<runtime-bootstrap>", "exec")
    payload = base64.b64encode(zlib.compress(source, level=9)).decode("ascii")
    digest = hashlib.sha256(source).hexdigest()
    wrapper = (
        "import base64, hashlib, zlib\n"
        f"source = zlib.decompress(base64.b64decode({payload!r}, validate=True))\n"
        f"if hashlib.sha256(source).hexdigest() != {digest!r}:\n"
        "    raise RuntimeError('Runtime bootstrap SHA256 mismatch')\n"
        "exec(compile(source, '<verified-runtime-bootstrap>', 'exec'))\n"
    )
    command = (
        f"cd {shlex.quote(str(working_directory))} && "
        f"exec {shlex.quote(str(python))} -u - <<'HWAVJEPA_BOOTSTRAP'\n"
        f"{wrapper}HWAVJEPA_BOOTSTRAP\n"
    )
    if len(command.encode("utf-8")) + 1 > MAX_COMMAND_BYTES:
        raise ValueError("Packaged command exceeds 64 KiB; put the overlay in the image/context")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--working-directory", default="/app")
    args = parser.parse_args()
    command = package_bootstrap(args.source.read_bytes(), working_directory=args.working_directory)
    args.output.write_text(command)
    print(f"Packaged command: {len(command.encode('utf-8'))} bytes")


if __name__ == "__main__":
    main()

"""Download one pinned BRIR archive with the official Hugging Face/Xet client.

The caller supervises this child process's wall time and scratch allowance, then
checks the publisher's byte count and SHA-256 before converting anything. All
download metadata/cache files live below --local-dir; the Xet chunk cache is off.
This helper has no storage writer and accepts neither arbitrary URLs nor revisions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

BRIR_REPOSITORY = "labhamlet/GRAM-Naturalistic-Scenes-Binaural"
BRIR_REVISION = "546f796991721befaffce60152af59bfd9ad64ff"
SOURCE_PATH = re.compile(r"train/webdataset_output/shard-0000[0-6][0-9]\.tar")


def download_brir(source_path: str, local_dir: Path, *, downloader=None) -> Path:
    if not isinstance(source_path, str) or SOURCE_PATH.fullmatch(source_path) is None:
        raise ValueError("Expected one of the seventy pinned training BRIR archive paths")
    local_dir = Path(local_dir)
    if not local_dir.is_absolute() or local_dir.is_symlink():
        raise ValueError("BRIR scratch directory must be an absolute, non-symlink path")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_dir = local_dir.resolve()
    home = local_dir / ".hf-home"
    # Set before importing the Hub package: its environment configuration is
    # captured at import. These are process-local settings in the supervised child.
    os.environ.update(
        {
            "HF_HOME": str(home),
            "HF_HUB_CACHE": str(home / "hub"),
            "HF_XET_CACHE": str(home / "xet"),
            "HF_XET_LOG_DEST": str(home / "xet/logs") + "/",
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "HF_XET_SHARD_CACHE_SIZE_LIMIT": "67108864",
            "HF_XET_LOG_DIR_MAX_SIZE": "16mb",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_VERBOSITY": "error",
            "HF_DEBUG": "0",
            "HF_HUB_DISABLE_XET": "0",
            "HF_HUB_ETAG_TIMEOUT": "30",
            "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        }
    )
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    downloaded = Path(
        downloader(
            repo_id=BRIR_REPOSITORY,
            repo_type="dataset",
            revision=BRIR_REVISION,
            filename=source_path,
            local_dir=str(local_dir),
            cache_dir=str(home / "hub"),
            token=False,
            etag_timeout=30,
        )
    )
    expected = local_dir / source_path
    if (
        downloaded != expected
        or not downloaded.is_file()
        or downloaded.is_symlink()
        or downloaded.resolve() != expected
    ):
        raise ValueError("BRIR downloader returned a path outside its single-file scratch contract")
    return downloaded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    path = download_brir(args.source_path, args.local_dir)
    print(json.dumps({"path": str(path)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Hub exceptions may include signed download URLs. The parent records
        # the exit code/source ID and controls bounded retries without those URLs.
        print(f"Pinned BRIR download failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None

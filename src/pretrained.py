"""Download the pinned official WavJEPA tensors without executing Hub Python code."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

OFFICIAL_REPOSITORY = "labhamlet/wavjepa-base"
OFFICIAL_REVISION = "6be4a5093f6c9adbbd55e00b6e5b8f067aa03345"
OFFICIAL_FILENAME = "model.safetensors"
OFFICIAL_SHA256 = "988546976c453353c14ebb0798f275ea5eb95270d75eed506ea455c5b49f6be0"


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download_official_weights(
    *, local_files_only: bool = False, cache_dir: str | Path | None = None
) -> Path:
    """Resolve one immutable, digest-verified safetensors file in the HF cache.

    This does not download/import the repository's Python modules or interpret
    its config (which contains a Python expression). Set local_files_only for
    offline runs; a missing cached file raises huggingface_hub's usual error.
    """
    if type(local_files_only) is not bool:
        raise ValueError("local_files_only must be boolean")
    path = Path(
        hf_hub_download(
            repo_id=OFFICIAL_REPOSITORY,
            filename=OFFICIAL_FILENAME,
            revision=OFFICIAL_REVISION,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    )
    if file_sha256(path) != OFFICIAL_SHA256:
        raise ValueError("official WavJEPA weight file failed SHA-256 verification")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    print(
        download_official_weights(local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    )


if __name__ == "__main__":
    main()

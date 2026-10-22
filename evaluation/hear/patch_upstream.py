"""Apply two checksum-verified compatibility fixes to the pinned HEAR install."""

import hashlib
import importlib.util
import tempfile
from pathlib import Path

UPSTREAM_COMMIT = "855964977238e89dfc76394aa11c37010edb6f20"


def _patch_file(source: Path, original_sha: str, patched_sha: str, replacements: list) -> None:
    contents = source.read_bytes()
    # Exact checksums prevent applying this compatibility patch to other code.
    digest = hashlib.sha256(contents).hexdigest()
    if digest == patched_sha:
        return
    if digest != original_sha:
        raise ValueError(f"HEAR source differs from pinned {UPSTREAM_COMMIT}; refusing patch")
    for old, new in replacements:
        contents = contents.replace(old, new)
    if hashlib.sha256(contents).hexdigest() != patched_sha:
        raise ValueError("Unexpected HEAR patch result")
    # Replace rather than truncate: uv may hard-link installed files to its cache.
    with tempfile.NamedTemporaryFile(dir=source.parent, delete=False) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    temporary_path.replace(source)


def patch() -> Path:
    spec = importlib.util.find_spec("heareval")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the pinned HEAR package with --no-deps first")
    root = Path(spec.origin).parent
    source = root / "embeddings" / "task_embeddings.py"
    _patch_file(
        source,
        "5f52217a07fb0d0bb4eaf44ceb41f4e77d0191d636138bdbfb3ed69b2458021b",
        "8f9bf399e52a690cbca388ba5c1475986b91ab103cb79711f71591ab90268508",
        [
            (
                b"import tensorflow as tf\n",
                b"try:\n    import tensorflow as tf\nexcept ImportError:\n"
                b"    tf = None  # PyTorch-only evaluation does not require TensorFlow.\n",
            ),
            (
                b"elif isinstance(self.model, tf.Module):",
                b"elif tf is not None and isinstance(self.model, tf.Module):",
            ),
        ],
    )
    _patch_file(
        root / "predictions" / "task_predictions.py",
        "bc4cd5e5ad0ecaf460b462f1141c94529c2ef552f1ca1bab526d6be1b719ee4b",
        "e858afdf09682806283e3ce33f6ecb9d7a3f0ef8b46d1a6b58f3799718dee59b",
        [
            (
                b"trainer.fit_loop.current_epoch = grid_point.epoch",
                b"trainer.fit_loop.epoch_progress.current.completed = grid_point.epoch",
            ),
        ],
    )
    return source


if __name__ == "__main__":
    print(patch())

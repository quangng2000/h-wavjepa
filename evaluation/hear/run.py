"""Package exported embeddings and call the official HEAR prediction runner.

Run only in the isolated evaluation environment described in docs/HEAR.md.
No downstream metric, probe model, split, or model-selection logic is replaced.
"""

import argparse
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
from patch_upstream import UPSTREAM_COMMIT, patch


def _child(root: Path, name: str) -> Path:
    value = (root / name).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("Export path escapes its directory")
    return value


def prepare(directory: Path) -> None:
    patched = patch()
    from heareval.embeddings.task_embeddings import (
        get_labels_for_timestamps,
        memmap_embeddings,
    )

    manifest = json.loads((directory / "h-wavjepa-export.json").read_text())
    if manifest.get("protocol_version") != 1:
        raise ValueError("Unsupported H-WavJEPA HEAR export protocol")
    if (directory / ".done.embeddings").exists():
        return
    metadata = json.loads((directory / "task_metadata.json").read_text())
    prng = random.Random(0)  # Same ordering as upstream task_embeddings.
    for split in metadata["splits"]:
        split_path = _child(directory, split)
        split_data = json.loads(_child(directory, f"{split}.json").read_text())
        for name in split_data:
            prefix = _child(split_path, name)
            values = np.load(f"{prefix}.embedding.npy", allow_pickle=False)
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Expected finite float32 exported embeddings")
            labels = json.loads(Path(f"{prefix}.raw-labels.json").read_text())
            if metadata["embedding_type"] == "event":
                timestamps = np.asarray(json.loads(Path(f"{prefix}.timestamps.json").read_text()))
                if not np.isfinite(timestamps).all() or len(timestamps) != len(values):
                    raise ValueError("Invalid timestamp export")
                labels = get_labels_for_timestamps([labels], timestamps[None])[0]
            Path(f"{prefix}.target-labels.json").write_text(json.dumps(labels) + "\n")
        memmap_embeddings(split_path, prng, metadata, split, directory, split_data)
    (directory / "h-wavjepa-evaluator.json").write_text(
        json.dumps(
            {
                "upstream_repository": "https://github.com/hearbenchmark/hear-eval-kit",
                "upstream_commit": UPSTREAM_COMMIT,
                "embedding_module_sha256": hashlib.sha256(patched.read_bytes()).hexdigest(),
                "prediction_module_sha256": hashlib.sha256(
                    (patched.parents[1] / "predictions" / "task_predictions.py").read_bytes()
                ).hexdigest(),
                "compatibility_patches": [
                    "optional TensorFlow import for PyTorch-only packaging",
                    "Lightning 1.9 best-epoch event postprocessing lookup",
                ],
                "runtime_versions": {
                    name: version(name)
                    for name in ("heareval", "torch", "pytorch-lightning", "numpy", "sed-eval")
                },
            },
            indent=2,
        )
        + "\n"
    )
    (directory / ".done.embeddings").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="+", type=Path)
    parser.add_argument("--grid", choices=("default", "fast", "faster"), default="default")
    parser.add_argument("--grid-points", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", default="null", help='HEAR GPU JSON; "null" means CPU')
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.grid_points < 1:
        parser.error("workers must be nonnegative; grid-points must be positive")
    directories = [directory.resolve(strict=True) for directory in args.exports]
    for directory in directories:
        prepare(directory)
    if not args.prepare_only:
        from heareval.predictions import task_predictions
        from heareval.predictions.runner import runner

        # Bound process spawning for portable local evaluation; the probe and
        # metrics remain upstream. Increase explicitly on larger machines.
        task_predictions.NUM_WORKERS = args.workers
        runner.main(
            args=[
                *map(str, directories),
                "--grid",
                args.grid,
                "--grid-points",
                str(args.grid_points),
                "--gpus",
                args.gpus,
            ],
            standalone_mode=False,
        )


if __name__ == "__main__":
    main()

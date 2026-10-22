"""Export HEAR task features in the training environment for official evaluation.

Only embedding extraction lives here. Label alignment, memmap packaging, probe
training, validation model selection, and scoring use the pinned official kit in
evaluation/hear. Process recordings individually so variable lengths are never
silently batch padded. This command does not resample or mix stereo channels.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from src.hear_api import (
    REPRESENTATIONS,
    HEARModel,
    get_scene_embeddings,
    get_timestamp_embeddings,
    load_model,
)
from src.utils.runtime import file_sha256

PROTOCOL_VERSION = 1


def _relative_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("HEAR file names must be nonempty relative paths")
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"HEAR path escapes its task directory: {name}")
    return candidate


def export_task(
    model: HEARModel,
    task_dir: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Write per-file arrays consumed by the official kit's packaging functions."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("HEAR export requires soundfile; install the 'hear' extra") from exc

    task = Path(task_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output}")
    metadata = json.loads((task / "task_metadata.json").read_text())
    if metadata.get("embedding_type") not in {"scene", "event"}:
        raise ValueError("HEAR embedding_type must be scene or event")
    splits = metadata.get("splits")
    if not isinstance(splits, list) or not splits or len(set(splits)) != len(splits):
        raise ValueError("HEAR metadata must contain distinct nonempty splits")
    for split in splits:
        if not isinstance(split, str) or Path(split).name != split or split in {".", ".."}:
            raise ValueError("HEAR split names must be single directory names")
    output.mkdir(parents=True)
    shutil.copy2(task / "task_metadata.json", output)
    shutil.copy2(task / "labelvocabulary.csv", output)
    for split in splits:
        labels = json.loads((task / f"{split}.json").read_text())
        if not isinstance(labels, dict) or not labels:
            raise ValueError(f"HEAR split {split} must contain file-to-label mappings")
        shutil.copy2(task / f"{split}.json", output)
        audio_root = task / str(model.sample_rate) / split
        for name, target in labels.items():
            source = _relative_file(audio_root, name)
            samples, rate = sf.read(source, dtype="float32", always_2d=False)
            if rate != model.sample_rate or samples.ndim != 1 or not len(samples):
                raise ValueError(f"{source} must be nonempty mono audio at {model.sample_rate} Hz")
            audio = torch.from_numpy(samples)[None]
            destination = _relative_file(output / split, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if metadata["embedding_type"] == "scene":
                embeddings = get_scene_embeddings(audio, model)[0]
            else:
                embeddings, timestamps = get_timestamp_embeddings(audio, model)
                embeddings = embeddings[0]
                Path(f"{destination}.timestamps.json").write_text(
                    json.dumps(timestamps[0].tolist()) + "\n"
                )
            np.save(f"{destination}.embedding.npy", embeddings.numpy(), allow_pickle=False)
            # Event labels remain intervals until the official kit aligns them.
            Path(f"{destination}.raw-labels.json").write_text(json.dumps(target) + "\n")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "module": "src.hear_api",
        "representation": model.representation,
        "sample_rate": model.sample_rate,
        "window_samples": model.encoder.window_samples,
        "window_batch_size": model.window_batch_size,
        "checkpoint_sha256": file_sha256(Path(checkpoint_path)) if checkpoint_path else None,
        "source_task": str(task),
        "temporal_grid": "native_tokens" if model.representation == "backbone" else "dense_windows",
        "timestamp_hop_samples": (
            model.encoder.backbone.stride_samples
            if model.representation == "backbone"
            else model.encoder.window_samples
        ),
        "training_state_strides": list(model.encoder.state_strides),
        "frontend_receptive_field_samples": model.encoder.backbone.receptive_field_samples,
        "padding": (
            "right-zero-pad context; retain only tokens with real frontend support; "
            "one centered padded token below one receptive field"
            if model.representation == "backbone"
            else "right-zero-pad scene tail and duration-weight; omit timestamp window tail "
            "unless clip is short, then return one centered padded-context state"
        ),
        "timestamp_context": "bidirectional completed window; not streaming",
    }
    (output / "h-wavjepa-export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="low")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--window-batch-size", type=int, default=8)
    args = parser.parse_args()
    model = load_model(
        args.checkpoint,
        representation=args.representation,
        device=args.device,
        window_batch_size=args.window_batch_size,
    )
    print(export_task(model, args.task_dir, args.output_dir, checkpoint_path=args.checkpoint))


if __name__ == "__main__":
    main()

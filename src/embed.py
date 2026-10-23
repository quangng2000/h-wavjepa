"""Stream mean-pooled audio representations into bounded-size PyTorch shards."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.wav_datamodule import WAVManifestDataset, _manifest_entries
from src.dynamics_inference import load_model as load_dynamics_model
from src.inference import load_model
from src.utils.runtime import atomic_json, file_sha256


def extract_embeddings(
    checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
    *,
    num_samples: int,
    batch_size: int = 16,
    device: str = "cpu",
    model_family: str = "legacy",
) -> dict:
    """Extract one vector per hierarchy level/file, plus the protected vector.

    The Python API retains its legacy default for existing callers; the CLI
    defaults to the active dynamics family. Pass model_family explicitly in new
    code. Temporal mean pooling does not guarantee preservation of token-level
    information.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    manifest = Path(manifest).expanduser().resolve(strict=True)
    if model_family not in {"legacy", "dynamics"}:
        raise ValueError("model_family must be legacy or dynamics")
    model = (load_dynamics_model if model_family == "dynamics" else load_model)(
        checkpoint, device=device
    )
    sample_rate = (
        model.backbone.sample_rate
        if model_family == "dynamics"
        else model.hparams.input_sample_rate
    )
    if sample_rate is None:
        raise ValueError("Checkpoint has no input sample rate; use one created by src.train")
    divisor = (
        model.window_samples
        if model_family == "dynamics"
        else model.encoder.patch_size * model.encoder.total_pool_factor
    )
    if type(num_samples) is not int or num_samples < divisor or num_samples % divisor:
        raise ValueError(f"num_samples must be a positive multiple of {divisor}")
    if model_family == "dynamics" and any(
        len(positions) == 0
        for positions in model.level_positions(num_samples // divisor, subsample=True)
    ):
        raise ValueError("num_samples does not contain a complete state at every training stride")
    entries = _manifest_entries(manifest)
    dataset = WAVManifestDataset(
        entries,
        sample_rate=sample_rate,
        num_samples=num_samples,
        require_full_length=model_family == "dynamics",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "model_family": model_family,
        "status": "running",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest),
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "pooling": "temporal_mean",
        "temporal_grid": "training_strides" if model_family == "dynamics" else "encoder_tokens",
        "level_widths": [model.backbone.embedding_dim] * 3
        if model_family == "dynamics"
        else list(model.encoder.widths),
        "examples": len(dataset),
        "shards": 0,
    }
    if model_family == "dynamics":
        record.update(
            input_channels=model.backbone.config.get("input_channels", 1),
            input_adaptation="mono duplicated to both ears"
            if model.backbone.config.get("input_channels", 1) == 2
            else "mono",
            aggregation_windows=list(model.aggregation_windows),
            level_support_windows=list(model.level_support_windows),
            training_state_strides=list(model.state_strides),
            temporal_context="whole manifest crop; never split or reset between internal windows",
            state_padding="none; only valid states at training-grid endpoints",
            level_endpoints=[
                positions.tolist()
                for positions in model.level_positions(num_samples // divisor, subsample=True)
            ],
        )
    atomic_json(directory / "metadata.json", record)
    offset = 0
    try:
        with (directory / "index.jsonl").open("x", encoding="utf-8") as index:
            with torch.inference_mode():
                for number, batch in enumerate(loader):
                    waveform = batch["waveform"].to(device)
                    if (
                        model_family == "dynamics"
                        and model.backbone.config.get("input_channels", 1) == 2
                    ):
                        # HEAR-compatible inference on clean mono sources only;
                        # this duplication is never used for Nat pretraining.
                        waveform = waveform[:, None].expand(-1, 2, -1)
                    encoded = (
                        {"levels": model.encode(waveform, subsample=True), "protected": None}
                        if model_family == "dynamics"
                        else model.encode(waveform)
                    )
                    shard = {
                        "levels": [level.mean(dim=1).cpu() for level in encoded["levels"]],
                        "protected": encoded["protected"].mean(dim=1).cpu()
                        if encoded["protected"] is not None
                        else None,
                    }
                    tensors = [
                        *shard["levels"],
                        *([shard["protected"]] if shard["protected"] is not None else []),
                    ]
                    if any(not torch.isfinite(value).all() for value in tensors):
                        raise FloatingPointError(
                            "Nonfinite embedding; no completed output was published"
                        )
                    name = f"batch-{number:06d}.pt"
                    temporary = directory / f".{name}.partial"
                    torch.save(shard, temporary)
                    temporary.replace(directory / name)
                    size = batch["waveform"].shape[0]
                    for row, entry in enumerate(entries[offset : offset + size]):
                        index.write(
                            json.dumps(
                                {
                                    "path": str(entry.path),
                                    "offset_samples": entry.offset_samples,
                                    "segment_samples": entry.segment_samples,
                                    "shard": name,
                                    "row": row,
                                }
                            )
                            + "\n"
                        )
                    offset += size
                    record["shards"] += 1
        record["status"] = "completed"
        atomic_json(directory / "metadata.json", record)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(directory / "metadata.json", record)
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory (must not exist)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Clip length; use the training length for comparable features",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-family",
        choices=("dynamics", "legacy"),
        default="dynamics",
        help="Checkpoint family; dynamics uses the official WavJEPA backbone and training strides",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    options = vars(args)
    options.pop("num_threads")
    print(json.dumps(extract_embeddings(**options), indent=2))


if __name__ == "__main__":
    main()

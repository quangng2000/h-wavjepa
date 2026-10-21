"""Stream mean-pooled audio representations into bounded-size PyTorch shards."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.wav_datamodule import WAVManifestDataset, _manifest_paths
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
) -> dict:
    """Extract one vector per hierarchy level/file, plus the protected vector.

    Temporal mean pooling is a downstream readout choice, not a guarantee that
    the pooled vector preserves the token-level protected information.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    manifest = Path(manifest).expanduser().resolve(strict=True)
    model = load_model(checkpoint, device=device)
    sample_rate = model.hparams.input_sample_rate
    if sample_rate is None:
        raise ValueError("Checkpoint has no input sample rate; use one created by src.train")
    divisor = model.encoder.patch_size * model.encoder.total_pool_factor
    if type(num_samples) is not int or num_samples < divisor or num_samples % divisor:
        raise ValueError(f"num_samples must be a positive multiple of {divisor}")
    paths = _manifest_paths(manifest)
    dataset = WAVManifestDataset(paths, sample_rate=sample_rate, num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "status": "running",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest),
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "pooling": "temporal_mean",
        "level_widths": list(model.encoder.widths),
        "examples": len(dataset),
        "shards": 0,
    }
    atomic_json(directory / "metadata.json", record)
    offset = 0
    try:
        with (directory / "index.jsonl").open("x", encoding="utf-8") as index:
            with torch.inference_mode():
                for number, batch in enumerate(loader):
                    encoded = model.encode(batch["waveform"].to(device))
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
                    for row, path in enumerate(paths[offset : offset + size]):
                        index.write(
                            json.dumps({"path": str(path), "shard": name, "row": row}) + "\n"
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

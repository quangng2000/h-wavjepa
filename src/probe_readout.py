"""Measure frozen official WavJEPA window readouts on a bounded local WAV cohort.

Run with ``python -m src.probe_readout --help``. This command never downloads
weights/audio or trains a hierarchy. Its variances describe the selected cohort,
not semantic usefulness or a proof of collapse in a subsequently trained model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.data.wav_datamodule import WAVManifestDataset, _manifest_entries
from src.models.components.wavjepa_backbone import UPSTREAM_COMMIT, WavJEPABackbone
from src.pretrained import (
    OFFICIAL_FILENAME,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    OFFICIAL_SHA256,
    file_sha256,
)


class CohortMoments:
    """Merge float64 moments over recordings, with equal weight per recording.

    All recordings must have the same number of windows. Between variance is
    computed over the WHOLE selected cohort, including differences between batch
    means. It is not an average of per-batch between-recording variances.
    """

    def __init__(self) -> None:
        self.count = 0
        self.shape: tuple[int, int] | None = None
        self.mean: Tensor | None = None
        self.between_m2: Tensor | None = None
        self.within_sum = 0.0
        self.norm_sum = 0.0
        self.adjacent_change_sum = 0.0

    def update(self, states: Tensor) -> None:
        if states.ndim != 3 or min(states.shape) < 1 or states.shape[1] < 2:
            raise ValueError("states must be [recording, time >= 2, feature]")
        values = states.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("states must contain only finite values")
        shape = tuple(values.shape[1:])
        if self.shape is not None and self.shape != shape:
            raise ValueError("all recordings must use the same window count and feature width")
        self.shape = shape
        count = values.shape[0]
        means = values.mean(dim=1)
        mean = means.mean(dim=0)
        m2 = (means - mean).square().sum(dim=0)
        if self.mean is None:
            self.mean, self.between_m2 = mean, m2
        else:
            delta = mean - self.mean
            self.between_m2 += m2 + delta.square() * (self.count * count / (self.count + count))
            self.mean += delta * (count / (self.count + count))
        self.within_sum += values.var(dim=1, correction=0).mean(dim=1).sum().item()
        self.norm_sum += values.norm(dim=-1).mean(dim=1).sum().item()
        self.adjacent_change_sum += (
            (values[:, 1:] - values[:, :-1]).square().mean(dim=(1, 2)).sum().item()
        )
        self.count += count

    def report(self) -> dict[str, Any]:
        if self.count < 2:
            raise ValueError("between-recording variance requires at least two recordings")
        within = self.within_sum / self.count
        between = (self.between_m2 / self.count).mean().item()
        total = within + between
        return {
            "recordings": self.count,
            "windows_per_recording": self.shape[0],
            "feature_width": self.shape[1],
            "variance_convention": "population; mean over features; equal recording weights",
            "between_population": "all selected recording means, merged across batches",
            "within_recording_variance": within,
            "between_recording_variance": between,
            "total_variance": total,
            "within_variance_fraction": within / total if total else 0.0,
            "within_recording_rms_std": math.sqrt(within),
            "between_recording_rms_std": math.sqrt(between),
            "mean_window_l2_norm": self.norm_sum / self.count,
            "adjacent_window_change_mse": self.adjacent_change_sum / self.count,
        }


def select_recordings(entries: list, max_recordings: int, seed: int) -> list[int]:
    """Use the first segment per recording, then a seeded sample in manifest order.

    An explicit recording_id groups files; physical file identity also prevents
    different segments/aliases of the same WAV from being counted as independent
    recordings. No new random crop is introduced by batching or inference.
    """
    if type(max_recordings) is not int or max_recordings < 2:
        raise ValueError("max_recordings must be an integer >= 2")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    # Resolve the whole equivalence relation before selecting representatives.
    # Otherwise a later alias linking two files could count the same recording
    # twice, depending on the order of its manifest segments.
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            next_key = parents[key]
            parents[key] = root
            key = next_key
        return root

    for entry in entries:
        physical = find(("file", *entry.identity))
        if entry.recording_id is not None:
            parents[physical] = find(("recording_id", entry.recording_id))
    seen, candidates = set(), []
    for index, entry in enumerate(entries):
        identity = find(("file", *entry.identity))
        if identity not in seen:
            seen.add(identity)
            candidates.append(index)
    if len(candidates) < 2:
        raise ValueError("manifest must contain at least two distinct recordings")
    return sorted(random.Random(seed).sample(candidates, min(max_recordings, len(candidates))))


@torch.inference_mode()
def encode_window_batches(
    backbone: WavJEPABackbone, waveforms: Tensor, window_batch_size: int, device: torch.device
) -> Tensor:
    """Bound encoder memory independently of the number of audio recordings."""
    if type(window_batch_size) is not int or window_batch_size < 1:
        raise ValueError("window_batch_size must be a positive integer")
    flat = waveforms.flatten(0, 1)
    states = [
        backbone.encode_windows(chunk.to(device).unsqueeze(1), readout_mode="normalized_top_k")
        .squeeze(1)
        .cpu()
        for chunk in flat.split(window_batch_size)
    ]
    return torch.cat(states).reshape(waveforms.shape[0], waveforms.shape[1], -1)


def publish_report(path: Path, report: dict[str, Any]) -> None:
    """Publish a complete small JSON atomically, refusing any existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link creation is atomic and fails if path exists (including a
        # racing writer); unlike replace(), it cannot overwrite another report.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_probe(
    *,
    weights: Path,
    manifest: Path,
    output: Path,
    max_recordings: int = 128,
    batch_size: int = 4,
    window_batch_size: int = 8,
    windows: int = 9,
    observed_windows: int = 5,
    window_samples: int = 16160,
    seed: int = 0,
    device: str = "cpu",
    num_threads: int = 4,
) -> dict[str, Any]:
    for name, value, minimum in (
        ("batch_size", batch_size, 1),
        ("window_batch_size", window_batch_size, 1),
        ("windows", windows, 2),
        ("observed_windows", observed_windows, 2),
        ("window_samples", window_samples, 1),
        ("num_threads", num_threads, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if observed_windows > windows:
        raise ValueError("observed_windows must not exceed windows")
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"report already exists: {output}")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda[:index]")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    weights = Path(weights).expanduser().resolve(strict=True)
    manifest = Path(manifest).expanduser().resolve(strict=True)
    weights_sha = file_sha256(weights)
    if weights_sha != OFFICIAL_SHA256:
        raise ValueError("weights do not match the pinned official WavJEPA SHA-256")
    manifest_sha = file_sha256(manifest)
    entries = _manifest_entries(manifest)
    indices = select_recordings(entries, max_recordings, seed)
    selected = [entries[index] for index in indices]
    dataset = WAVManifestDataset(
        selected,
        sample_rate=16000,
        num_samples=windows * window_samples,
        require_full_length=True,
        random_crop=False,
        seed=seed,
    )
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    backbone = WavJEPABackbone(
        config={"window_samples": window_samples},
        pretrained_path=str(weights),
        local_files_only=True,
    )
    if backbone.checkpoint_sha256 != weights_sha:
        raise ValueError("weights changed while loading the official checkpoint")
    backbone.requires_grad_(False).eval().to(target)
    moments = CohortMoments()
    observed_moments = CohortMoments()
    for start in range(0, len(dataset), batch_size):
        audio = torch.stack(
            [
                dataset[index]["waveform"]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
        ).reshape(-1, windows, window_samples)
        states = encode_window_batches(backbone, audio, window_batch_size, target)
        moments.update(states)
        observed_moments.update(states[:, :observed_windows])
    if file_sha256(manifest) != manifest_sha:
        raise ValueError("manifest changed during the readout probe")
    report = {
        "schema_version": 1,
        "status": "complete",
        "probe": "frozen_official_wavjepa_normalized_top_k_window_readout",
        "initialization": {
            "kind": "official_released_weights",
            "repository": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REVISION,
            "filename": OFFICIAL_FILENAME,
            "weights_sha256": weights_sha,
            "official_sha256_verified": True,
            "vendored_training_core_commit": UPSTREAM_COMMIT,
            "fixed_positions": "regenerated only when window length differs from release",
        },
        "readout": {
            "mode": "normalized_top_k",
            "encoder": "online encoder from official release; eval; frozen; inference_mode",
            "top_k_layers": 8,
            "pooling": "mean over token positions after upstream top-k normalization",
            "hierarchy": False,
        },
        "cohort": {
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha,
            "manifest_entries": len(entries),
            "max_recordings": max_recordings,
            "seed": seed,
            "subset_policy": "first segment per recording_id/physical file; seeded uniform subset; manifest order",
            "crop_policy": "fixed manifest offset; contiguous windows; no padding or resampling",
            "audio_payload_hashes_verified": False,
            "samples": [
                {
                    "manifest_entry_index": index,
                    "manifest_location": entry.location,
                    "recording_id": entry.recording_id,
                    "sample_id": entry.recording_id or str(entry.path),
                    "path": str(entry.path),
                    "offset_samples": entry.offset_samples,
                    "segment_samples": dataset.segment_frame_counts[position],
                }
                for position, (index, entry) in enumerate(zip(indices, selected, strict=True))
            ],
        },
        "window_geometry": {
            "sample_rate": 16000,
            "windows": windows,
            "observed_prefix_windows": observed_windows,
            "window_samples": window_samples,
            "stride_samples": window_samples,
            "window_seconds": window_samples / 16000,
            "total_real_samples": windows * window_samples,
            "total_seconds": windows * window_samples / 16000,
        },
        "execution": {
            "device": str(target),
            "dtype": "float32",
            "recording_batch_size": batch_size,
            "window_batch_size": window_batch_size,
            "num_threads": num_threads,
            "torch_version": str(torch.__version__),
        },
        "statistics": moments.report(),
        "observed_prefix_statistics": observed_moments.report(),
        "interpretation": "Cohort descriptive statistics only; no collapse threshold or semantic claim. Compare absolute variances with their ratio. Audio provenance depends on the supplied manifest.",
    }
    publish_report(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Local pinned official model.safetensors; SHA-256 required; never downloaded",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Local JSONL WAV manifest; explicit offsets are honored",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Fresh JSON output path; never overwritten"
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=128,
        help="Maximum distinct recordings, at least 2 (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Recordings decoded together (default: 4)"
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=8,
        help="Maximum completed windows per encoder call (default: 8)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=9,
        help="Contiguous nonoverlapping windows per recording, at least 2 (default: 9)",
    )
    parser.add_argument(
        "--observed-windows",
        type=int,
        default=5,
        help="Also report the initial observed prefix from the same states, from 2 through --windows (default: 5)",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=16160,
        help="Samples per 16 kHz window (default: 16160 = 1.01 seconds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Reproducible recording subset seed; crops remain at manifest offsets",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda[:index]; uses existing hardware only (default: cpu)",
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="PyTorch CPU threads (default: 4)"
    )
    args = parser.parse_args()
    report = run_probe(**vars(args))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "statistics": report["statistics"],
                "observed_prefix_statistics": report["observed_prefix_statistics"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

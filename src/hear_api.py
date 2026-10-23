"""HEAR 2021 API for hierarchical checkpoints, without evaluation dependencies.

The backbone is bidirectional within each independently normalized audio window.
Timestamps locate features; they do not claim streaming availability or narrow
context support. See docs/HEAR.md for padding and temporal-grid semantics.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.dynamics_inference import load_model as load_dynamics_model
from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule

REPRESENTATIONS = ("low", "medium", "high", "backbone")


class HEARModel(nn.Module):
    """A frozen encoder with bounded window batches and an explicit readout."""

    def __init__(
        self,
        encoder: HierarchicalWavJEPAModule,
        representation: str = "low",
        window_batch_size: int = 8,
    ) -> None:
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}")
        if type(window_batch_size) is not int or window_batch_size < 1:
            raise ValueError("window_batch_size must be a positive integer")
        self.encoder = encoder.eval().requires_grad_(False)
        self.representation = representation
        self.window_batch_size = window_batch_size
        self.sample_rate = encoder.backbone.sample_rate
        self.scene_embedding_size = encoder.backbone.embedding_dim
        self.timestamp_embedding_size = self.scene_embedding_size
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device


def load_model(
    model_file_path: str | Path,
    representation: str = "low",
    device: str = "cpu",
    window_batch_size: int = 8,
) -> HEARModel:
    """Load a local dynamics checkpoint via the explicit tensor-only loader.

    No model is downloaded implicitly and no checkpoint-controlled imports run.
    ``representation`` selects low/medium/high window states or backbone tokens.
    """
    return HEARModel(
        load_dynamics_model(model_file_path, device=device),
        representation=representation,
        window_batch_size=window_batch_size,
    )


def _validate_audio(audio: Tensor, model: HEARModel) -> None:
    if not isinstance(audio, Tensor) or audio.ndim != 2 or min(audio.shape) < 1:
        raise ValueError("audio must be a nonempty [sounds, samples] tensor")
    if audio.dtype != torch.float32 or not torch.isfinite(audio).all():
        raise ValueError("audio must contain finite float32 samples")
    if torch.any(audio.abs() > 1.0):
        raise ValueError("HEAR audio must be mono samples in [-1, 1]")
    if model.training or model.encoder.training:
        raise ValueError("HEAR extraction requires model.eval()")
    if model.device.type not in {"cpu", "cuda"}:
        raise ValueError("HEAR extraction currently supports CPU/CUDA")


def _window_readouts(audio: Tensor, model: HEARModel, starts: list[int]) -> Tensor:
    """Return CPU embeddings, never put the whole recording on the accelerator."""
    size = model.encoder.window_samples
    recordings = []
    for recording in audio:
        batches = []
        for first in range(0, len(starts), model.window_batch_size):
            batch_starts = starts[first : first + model.window_batch_size]
            windows = torch.stack(
                [
                    F.pad(
                        recording[start : start + size], (0, max(0, start + size - len(recording)))
                    )
                    for start in batch_starts
                ]
            ).to(model.device)
            # A batch of independent windows; no future-window activations mix.
            if model.representation == "backbone":
                values = model.encoder.backbone.encode_tokens(windows[:, None])[:, 0]
            else:
                low = model.encoder.backbone.encode_windows(
                    windows[:, None], readout_mode=model.encoder.readout_mode
                )
                levels = model.encoder._levels(low, model.encoder.abstractions)
                values = levels[REPRESENTATIONS.index(model.representation)][:, 0]
            batches.append(values.float().cpu())
        recordings.append(torch.cat(batches))
    return torch.stack(recordings)


def _token_embeddings(audio: Tensor, model: HEARModel) -> tuple[Tensor, Tensor]:
    backbone = model.encoder.backbone
    local_times = backbone.token_timestamps_ms().float().cpu()
    count = len(local_times)
    hop = count * backbone.stride_samples
    # Overlap only enough to retain a continuous frontend token grid between
    # independently encoded windows, rather than fabricating interpolated tokens.
    starts = list(range(0, audio.shape[1], hop))
    windows = _window_readouts(audio, model, starts)
    embeddings = windows.flatten(1, 2)
    times = (
        torch.tensor(starts, dtype=torch.float32)[:, None] * (1000.0 / model.sample_rate)
        + local_times[None]
    ).flatten()
    # Exclude tokens whose local frontend support extends into end padding.
    ends = (
        torch.arange(len(times), dtype=torch.long) * backbone.stride_samples
        + backbone.receptive_field_samples
    )
    valid = ends <= audio.shape[1]
    if valid.any():
        embeddings, times = embeddings[:, valid], times[valid]
    else:
        # HEAR accepts arbitrarily short clips. One padded context token is
        # unavoidable below the frontend receptive field; report the clip center.
        embeddings = embeddings[:, :1]
        times = torch.tensor([(audio.shape[1] - 1) * 500.0 / model.sample_rate])
    return embeddings, times.expand(audio.shape[0], -1).clone()


@torch.inference_mode()
def get_timestamp_embeddings(audio: Tensor, model: HEARModel) -> tuple[Tensor, Tensor]:
    """Return float32 CPU features and centered millisecond timestamps.

    Window spaces use complete nonoverlapping windows on the *dense* grid at
    every level; no upper-level interpolation is performed. A trailing partial
    window is omitted, except that a clip shorter than one window is right-padded
    once and represented at its real center. Backbone mode keeps valid tokens on
    the native frontend grid, using minimally overlapping context windows.
    """
    _validate_audio(audio, model)
    if model.representation == "backbone":
        return _token_embeddings(audio, model)
    size = model.encoder.window_samples
    complete = audio.shape[1] // size
    starts = [index * size for index in range(max(1, complete))]
    values = _window_readouts(audio, model, starts)
    support = min(size, audio.shape[1])
    times = (torch.tensor(starts, dtype=torch.float32) + (support - 1) / 2) * (
        1000.0 / model.sample_rate
    )
    return values, times.expand(audio.shape[0], -1).clone()


@torch.inference_mode()
def get_scene_embeddings(audio: Tensor, model: HEARModel) -> Tensor:
    """Pool all real audio; weight a right-padded last window by real duration."""
    _validate_audio(audio, model)
    if model.representation == "backbone":
        values, _ = _token_embeddings(audio, model)
        return values.mean(dim=1)
    size = model.encoder.window_samples
    starts = list(range(0, audio.shape[1], size))
    values = _window_readouts(audio, model, starts)
    weights = torch.tensor(
        [min(size, audio.shape[1] - start) for start in starts], dtype=torch.float32
    )
    return (values * weights[None, :, None]).sum(dim=1) / weights.sum()

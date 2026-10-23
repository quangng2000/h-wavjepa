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
    def support_windows(self) -> int:
        """Span, in low windows, of the selected dense state's causal support."""
        if self.representation == "backbone":
            return 1
        return self.encoder.level_support_windows[REPRESENTATIONS.index(self.representation)]

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
    channels = model.encoder.backbone.config.get("input_channels", 1)
    if not isinstance(audio, Tensor) or audio.ndim not in (2, 3) or min(audio.shape) < 1:
        raise ValueError("audio must be a nonempty [sounds, samples] tensor")
    if audio.ndim == 3 and (channels != 2 or audio.shape[1] != 2):
        raise ValueError("Stereo input requires a binaural model and exactly two channels")
    if audio.dtype != torch.float32 or not torch.isfinite(audio).all():
        raise ValueError("audio must contain finite float32 samples")
    if torch.any(audio.abs() > 1.0):
        raise ValueError("HEAR audio must be samples in [-1, 1]")
    if model.training or model.encoder.training:
        raise ValueError("HEAR extraction requires model.eval()")
    if model.device.type not in {"cpu", "cuda"}:
        raise ValueError("HEAR extraction currently supports CPU/CUDA")


def _level_readouts(low: Tensor, model: HEARModel) -> Tensor:
    """Aggregate CPU-cached low states with bounded accelerator lookback.

    Every chunk includes the full causal support of its first output. The low
    state cache is on CPU, so window batching never resets an upper state or
    copies the entire recording onto the accelerator. Only evaluation clips
    shorter than the selected support repeat their first state on the left.
    """
    support = model.support_windows
    index = REPRESENTATIONS.index(model.representation)
    if len(low) < support:
        low = torch.cat([low[:1].expand(support - len(low), -1), low])
    outputs = []
    for first in range(support - 1, len(low), model.window_batch_size):
        stop = min(first + model.window_batch_size, len(low))
        context = low[first - support + 1 : stop][None].to(model.device)
        values = model.encoder.levels_from_low(context)[index][0]
        outputs.append(values.float().cpu())
    return torch.cat(outputs)


def _window_readouts(audio: Tensor, model: HEARModel, starts: list[int]) -> Tensor:
    """Return valid dense CPU states using complete past-window support."""
    size = model.encoder.window_samples
    recordings = []
    for recording in audio:
        batches = []
        for first in range(0, len(starts), model.window_batch_size):
            batch_starts = starts[first : first + model.window_batch_size]
            windows = torch.stack(
                [
                    F.pad(
                        recording[..., start : start + size],
                        (0, max(0, start + size - recording.shape[-1])),
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
                values = low[:, 0]
            batches.append(values.float().cpu())
        values = torch.cat(batches)
        recordings.append(
            values if model.representation == "backbone" else _level_readouts(values, model)
        )
    return torch.stack(recordings)


def _token_embeddings(audio: Tensor, model: HEARModel) -> tuple[Tensor, Tensor]:
    backbone = model.encoder.backbone
    local_times = backbone.token_timestamps_ms().float().cpu()
    channels = backbone.config.get("input_channels", 1)
    if channels == 2:
        local_times = local_times[: len(local_times) // 2]
    count = len(local_times)
    hop = count * backbone.stride_samples
    # Overlap only enough to retain a continuous frontend token grid between
    # independently encoded windows, rather than fabricating interpolated tokens.
    starts = list(range(0, audio.shape[-1], hop))
    windows = _window_readouts(audio, model, starts)
    if channels == 2:
        # Both ear tokens describe the same time. Match their grid before
        # averaging; never treat the second ear as a later part of the clip.
        windows = windows.reshape(*windows.shape[:2], 2, count, windows.shape[-1]).mean(dim=2)
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
    valid = ends <= audio.shape[-1]
    if valid.any():
        embeddings, times = embeddings[:, valid], times[valid]
    else:
        # HEAR accepts arbitrarily short clips. One padded context token is
        # unavoidable below the frontend receptive field; report the clip center.
        embeddings = embeddings[:, :1]
        times = torch.tensor([(audio.shape[-1] - 1) * 500.0 / model.sample_rate])
    return embeddings, times.expand(audio.shape[0], -1).clone()


@torch.inference_mode()
def get_timestamp_embeddings(audio: Tensor, model: HEARModel) -> tuple[Tensor, Tensor]:
    """Return float32 CPU features and centered millisecond timestamps.

    Window spaces use the *dense valid* grid, centered over each state's full
    temporal support; no upper-level interpolation is performed. A trailing
    partial waveform window is omitted unless the clip is shorter than one
    window, when it is right-zero-padded. Clips with fewer low states than the
    selected upper support repeat the first low state on the left and return one
    state at the real clip center. This fallback is evaluation-only. Backbone
    mode retains valid tokens on its native frontend grid.
    """
    _validate_audio(audio, model)
    if model.encoder.backbone.config.get("input_channels", 1) == 2 and audio.ndim == 2:
        audio = audio[:, None].expand(-1, 2, -1)
    if model.representation == "backbone":
        return _token_embeddings(audio, model)
    size = model.encoder.window_samples
    complete = audio.shape[-1] // size
    starts = [index * size for index in range(max(1, complete))]
    values = _window_readouts(audio, model, starts)
    support = model.support_windows
    if len(starts) < support or complete == 0:
        times = torch.tensor([(audio.shape[-1] - 1) * 500.0 / model.sample_rate])
    else:
        # Dense state i spans [i, i + support) low windows, even though its
        # training-grid identity is the physical endpoint i + support - 1.
        times = (torch.arange(len(starts) - support + 1) * size + (support * size - 1) / 2) * (
            1000.0 / model.sample_rate
        )
    return values, times.expand(audio.shape[0], -1).clone()


@torch.inference_mode()
def get_scene_embeddings(audio: Tensor, model: HEARModel) -> Tensor:
    """Mean valid states, weighting each by its endpoint window's real duration.

    The final partial waveform window is right-zero-padded. For upper states,
    endpoint duration weights avoid repeatedly counting overlapping support as
    additional duration; an evaluation-only left repeat handles short clips.
    """
    _validate_audio(audio, model)
    if model.encoder.backbone.config.get("input_channels", 1) == 2 and audio.ndim == 2:
        audio = audio[:, None].expand(-1, 2, -1)
    if model.representation == "backbone":
        values, _ = _token_embeddings(audio, model)
        return values.mean(dim=1)
    size = model.encoder.window_samples
    starts = list(range(0, audio.shape[-1], size))
    values = _window_readouts(audio, model, starts)
    endpoint_starts = starts[model.support_windows - 1 :] or starts[-1:]
    weights = torch.tensor(
        [min(size, audio.shape[-1] - start) for start in endpoint_starts], dtype=torch.float32
    )
    return (values * weights[None, :, None]).sum(dim=1) / weights.sum()

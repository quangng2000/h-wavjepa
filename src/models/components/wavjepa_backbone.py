"""Small trainable adapter around the pinned official WavJEPA training core.

Windows are normalized and encoded independently. This makes a representation
available at its window END; its internal bidirectional tokens are not streaming
states. The original masked loss is retained as an auxiliary training objective.
"""

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.pretrained import download_official_weights, file_sha256
from src.vendor.wavjepa.extractors import ConvFeatureExtractor
from src.vendor.wavjepa.jepa import JEPA
from src.vendor.wavjepa.masking import TimeInverseBlockMasker
from src.vendor.wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

UPSTREAM_COMMIT = "6f4e1b1f18316c50768c8523f325ccffeaa6985b"

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "window_samples": 32160,
    "conv_layers": [[512, 10, 5]] + [[512, 3, 2]] * 4 + [[512, 2, 2]],
    "encoder_dim": 768,
    "encoder_heads": 12,
    "encoder_layers": 12,
    "decoder_dim": 384,
    "decoder_heads": 12,
    "decoder_layers": 12,
    "mlp_ratio": 4.0,
    "average_top_k_layers": 8,
    "compile_modules": False,
    "use_gradient_checkpointing": False,
    "masker": {
        "target_masks_per_context": 4,
        "context_mask_prob": 0.65,
        "context_mask_length": 10,
        "target_prob": 0.25,
        "target_length": 10,
        "ratio_cutoff": 0.1,
    },
}


class WavJEPABackbone(nn.Module):
    """Official model plus explicit, differentiable completed-window readout."""

    source_commit = UPSTREAM_COMMIT

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if type(pretrained) is not bool or type(local_files_only) is not bool:
            raise ValueError("pretrained and local_files_only must be boolean")
        if sum((checkpoint_path is not None, pretrained, pretrained_path is not None)) > 1:
            raise ValueError("choose only one of checkpoint_path, pretrained, pretrained_path")
        incoming = {} if config is None else dict(config)
        unknown = set(incoming) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown WavJEPA configuration fields: {sorted(unknown)}")
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg.update({k: v for k, v in incoming.items() if k != "masker"})
        masker = dict(incoming.get("masker", {}))
        if set(masker) - set(cfg["masker"]):
            raise ValueError("unknown WavJEPA masker configuration fields")
        cfg["masker"].update(masker)
        cfg["conv_layers"] = [list(spec) for spec in cfg["conv_layers"]]
        self._validate_config(cfg)
        self.config = cfg
        self.sample_rate = cfg["sample_rate"]
        self.window_samples = cfg["window_samples"]
        self.embedding_dim = cfg["encoder_dim"]
        self.checkpoint_sha256: str | None = None
        self.stride_samples, self.receptive_field_samples = 1, 1
        for _, kernel, stride in cfg["conv_layers"]:
            self.receptive_field_samples += (kernel - 1) * self.stride_samples
            self.stride_samples *= stride
        # The official base release has a fixed architecture and 200 positions.
        # Reject incompatible presets before any network request or allocation.
        if pretrained and any(
            cfg[key] != DEFAULT_CONFIG[key] for key in DEFAULT_CONFIG if key != "masker"
        ):
            raise ValueError("official pretrained base requires the default backbone architecture")
        extractor = ConvFeatureExtractor(
            conv_layers_spec=[tuple(spec) for spec in cfg["conv_layers"]],
            in_channels=1,
            depthwise=False,
        )
        self.model = JEPA(
            feature_extractor=extractor,
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(
                d_model=cfg["encoder_dim"],
                nhead=cfg["encoder_heads"],
                mlp_ratio=cfg["mlp_ratio"],
            ),
            transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=cfg["encoder_layers"]),
            transformer_decoder_layers_cfg=TransformerLayerCFG.create(
                d_model=cfg["decoder_dim"],
                nhead=cfg["decoder_heads"],
                mlp_ratio=cfg["mlp_ratio"],
            ),
            transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=cfg["decoder_layers"]),
            average_top_k_layers=cfg["average_top_k_layers"],
            resample_sr=self.sample_rate,
            process_audio_seconds=self.window_samples / self.sample_rate,
            target_length=self.window_samples,
            nr_samples_per_audio=1,
            compile_modules=False,
            use_gradient_checkpointing=False,
        )
        self.masker = TimeInverseBlockMasker(**cfg["masker"])
        if checkpoint_path is not None:
            self.load_official_checkpoint(checkpoint_path)
        elif pretrained_path is not None:
            self.load_huggingface_checkpoint(pretrained_path)
        elif pretrained:
            self.load_huggingface_checkpoint(
                download_official_weights(local_files_only=local_files_only)
            )

    @classmethod
    def from_pretrained(cls, *, local_files_only: bool = False) -> "WavJEPABackbone":
        """Load the pinned official base release without trust_remote_code."""
        return cls(pretrained=True, local_files_only=local_files_only)

    @staticmethod
    def _validate_config(cfg: dict[str, Any]) -> None:
        for key in (
            "sample_rate",
            "window_samples",
            "encoder_dim",
            "encoder_heads",
            "encoder_layers",
            "decoder_dim",
            "decoder_heads",
            "decoder_layers",
            "average_top_k_layers",
        ):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for prefix in ("encoder", "decoder"):
            width = cfg[f"{prefix}_dim"]
            if width % 2 or width % cfg[f"{prefix}_heads"]:
                raise ValueError(f"{prefix}_dim must be even and divisible by its head count")
        if not 1 <= cfg["average_top_k_layers"] <= cfg["encoder_layers"]:
            raise ValueError("average_top_k_layers must fit the encoder depth")
        if cfg["compile_modules"] is not False or cfg["use_gradient_checkpointing"] is not False:
            raise ValueError("compile_modules and use_gradient_checkpointing are unsupported")
        if not math.isfinite(cfg["mlp_ratio"]) or cfg["mlp_ratio"] < 1:
            raise ValueError("mlp_ratio must be finite and at least one")
        patches = cfg["window_samples"]
        if not cfg["conv_layers"]:
            raise ValueError("conv_layers must not be empty")
        for spec in cfg["conv_layers"]:
            if len(spec) != 3 or any(type(x) is not int or x < 1 for x in spec):
                raise ValueError("conv_layers must contain positive integer [width,kernel,stride]")
            patches = (patches - spec[1]) // spec[2] + 1
            if patches < 2:
                raise ValueError("window is too short for the convolutional frontend")
        mask = cfg["masker"]
        for key in ("target_masks_per_context", "context_mask_length", "target_length"):
            if type(mask[key]) is not int or mask[key] < 1:
                raise ValueError(f"masker {key} must be a positive integer")
        for key in ("context_mask_prob", "target_prob", "ratio_cutoff"):
            if not 0 < mask[key] < 1 or not math.isfinite(mask[key]):
                raise ValueError(f"masker {key} must be in (0,1)")
        for prob, length in (
            ("context_mask_prob", "context_mask_length"),
            ("target_prob", "target_length"),
        ):
            if mask[length] >= patches or mask[prob] * patches / mask[length] < 1:
                raise ValueError("mask span/probability is incompatible with the number of patches")

    def construction_config(self) -> dict[str, Any]:
        return deepcopy(self.config)

    def _normalized_windows(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim != 3 or waveforms.shape[-1] != self.window_samples:
            raise ValueError(f"expected waveform [B,T,{self.window_samples}]")
        if min(waveforms.shape) < 1 or not waveforms.is_floating_point():
            raise ValueError("waveforms must be nonempty floating-point tensors")
        if not torch.isfinite(waveforms).all():
            raise ValueError("waveforms must contain only finite values")
        flat = waveforms.reshape(-1, 1, self.window_samples).float()
        # Exactly the original per-crop mean/std transform, without random crops,
        # shuffling or forced bfloat16. Precision is owned by the outer trainer.
        return (flat - flat.mean(dim=(-2, -1), keepdim=True)) / (
            flat.std(dim=(-2, -1), keepdim=True) + 1e-5
        )

    def encode_tokens(self, waveforms: Tensor, teacher: bool = False) -> Tensor:
        """Return [batch, windows, tokens, width] with original positional embeddings.

        Each token is bidirectional within its completed window. Nominal token
        centers describe frontend sampling, not the contextual receptive field.
        The teacher path is detached, matching encode_teacher_windows.
        """
        if type(teacher) is not bool:
            raise ValueError("teacher must be boolean")
        if teacher:
            with torch.no_grad():
                return self._encode_tokens(waveforms, teacher=True)
        return self._encode_tokens(waveforms, teacher=False)

    def _encode_tokens(self, waveforms: Tensor, teacher: bool) -> Tensor:
        x = self._normalized_windows(waveforms)
        x = self.model.feature_norms(self.model.extract_audio(x))
        if self.model.post_extraction_mapper is not None:
            x = self.model.post_extraction_mapper(x)
        x = x + self.model.pos_encoding_encoder
        x = self.model._forward_teacher(x.detach()) if teacher else self.model.encoder_forward(x)
        return x.reshape(
            waveforms.shape[0], waveforms.shape[1], self.model.total_patches, self.embedding_dim
        )

    def token_timestamps_ms(self) -> Tensor:
        """Nominal centers of convolutional support, relative to window start."""
        samples = (
            torch.arange(self.model.total_patches, dtype=torch.float64) * self.stride_samples
            + (self.receptive_field_samples - 1) / 2
        )
        return samples * (1000.0 / self.sample_rate)

    def encode_windows(self, waveforms: Tensor) -> Tensor:
        """Encode each completed window, retaining gradients to the official encoder."""
        return self.encode_tokens(waveforms).mean(dim=2)

    def masked_loss(self, waveforms: Tensor, seed: int | None = None) -> Tensor:
        """Original upstream masked objective, evaluated independently per window."""
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be an integer in [0,2**63) or None")
        if seed is None:
            # Follow Lightning/PyTorch's seeded training stream rather than
            # NumPy default_rng(None), which would draw untracked OS entropy.
            seed = int(torch.randint(0, 2**63 - 1, (), device="cpu").item())
        audio = self._normalized_windows(waveforms)
        masks = self.masker(
            audio.shape[0],
            self.model.total_patches,
            1,
            generator=np.random.default_rng(seed),
        )
        masks = tuple(mask.to(audio.device) for mask in masks)
        if not masks[1].any() or not (~masks[0]).any(dim=1).all():
            raise RuntimeError("upstream masker produced an empty target or context")
        return self.model(audio, *masks)["loss"]

    @torch.no_grad()
    def encode_teacher_windows(self, waveforms: Tensor) -> Tensor:
        """Read out upstream normalized top-layer targets, without cross-window context.

        As in WavJEPA, only the context Transformer is EMA: frontend, feature norm
        and mapper are shared with the student and detached on this target path.
        """
        return self.encode_tokens(waveforms, teacher=True).mean(dim=2)

    @torch.no_grad()
    def update_masked_teacher(self, decay: float) -> None:
        """Original encoder-only EMA; outer training owns timing and decay schedule."""
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0,1)")
        for student, teacher in zip(
            self.model.encoder.parameters(), self.model.teacher_encoder.parameters(), strict=True
        ):
            teacher.mul_(decay).add_(student, alpha=1 - decay)

    def load_official_checkpoint(self, path: str | Path) -> None:
        """Load a trusted local original JEPA state dict; never unpickle arbitrary objects.

        Compiled-module key segments are removed explicitly and collision-checked.
        Hugging Face safetensors must use load_huggingface_checkpoint instead.
        """
        source = Path(path).expanduser().resolve(strict=True)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else None
        if (
            not isinstance(state, dict)
            or not state
            or not all(isinstance(k, str) and isinstance(v, Tensor) for k, v in state.items())
        ):
            raise ValueError("checkpoint must contain a tensor-only original JEPA state_dict")
        normalized = {}
        for key, value in state.items():
            clean_key = key.replace("._orig_mod.", ".")
            if clean_key in normalized:
                raise ValueError("compiled checkpoint keys collide after normalization")
            if not torch.isfinite(value).all():
                raise ValueError("checkpoint contains nonfinite model weights")
            normalized[clean_key] = value
        self._load_checked_state(normalized)
        self.checkpoint_sha256 = file_sha256(source)

    def load_huggingface_checkpoint(self, path: str | Path) -> None:
        """Load the explicit official HF export layout: tensor keys prefixed model.

        No arbitrary prefix guessing, positional resizing, missing-key fallback
        or remote Python execution is allowed. Local files may be used offline;
        only download_official_weights also verifies the published file digest.
        """
        from safetensors.torch import load_file

        source = Path(path).expanduser().resolve(strict=True)
        state = load_file(source, device="cpu")
        if not state or any(not key.startswith("model.") for key in state):
            raise ValueError("expected official Hugging Face safetensors with model. keys")
        normalized = {key.removeprefix("model."): value for key, value in state.items()}
        if len(normalized) != len(state):
            raise ValueError("Hugging Face checkpoint keys collide after normalization")
        self._load_checked_state(normalized)
        self.checkpoint_sha256 = file_sha256(source)

    def _load_checked_state(self, state: Mapping[str, Tensor]) -> None:
        """Validate every tensor before mutating any model parameter."""
        expected = self.model.state_dict()
        if set(state) != set(expected) or any(
            state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype
            for key in expected
        ):
            raise ValueError("checkpoint is incompatible with the configured original JEPA model")
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("checkpoint contains nonfinite model weights")
        self.model.load_state_dict(state, strict=True)

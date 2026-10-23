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
from src.vendor.wavjepa.extractors import ConvChannelFeatureExtractor, ConvFeatureExtractor
from src.vendor.wavjepa.jepa import JEPA
from src.vendor.wavjepa.masking import TimeInverseBlockMasker
from src.vendor.wavjepa.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_binaural_sincos_pos_embed,
)
from src.vendor.wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

UPSTREAM_COMMIT = "6f4e1b1f18316c50768c8523f325ccffeaa6985b"

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "input_channels": 1,
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
    "compact_masked_tokens": False,
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
        self.input_channels = cfg["input_channels"]
        self.window_samples = cfg["window_samples"]
        self.embedding_dim = cfg["encoder_dim"]
        self.checkpoint_sha256: str | None = None
        self.stride_samples, self.receptive_field_samples = 1, 1
        for _, kernel, stride in cfg["conv_layers"]:
            self.receptive_field_samples += (kernel - 1) * self.stride_samples
            self.stride_samples *= stride
        # The official base/Nat releases share fixed Transformer dimensions.
        # Reject incompatible presets before any network request or allocation.
        if pretrained and any(
            cfg[key] != DEFAULT_CONFIG[key]
            for key in DEFAULT_CONFIG
            if key
            not in {
                "masker",
                "input_channels",
                "window_samples",
                "use_gradient_checkpointing",
                "compact_masked_tokens",
            }
        ):
            raise ValueError(
                "official pretrained weights require the default backbone architecture"
            )
        extractor_kwargs = {
            "conv_layers_spec": [tuple(spec) for spec in cfg["conv_layers"]],
            "in_channels": self.input_channels,
            "depthwise": False,
        }
        if self.input_channels == 2:
            # Section 3.3 specifies two independent waveform encoders. The
            # pinned official channel extractor retains separate learned CNNs
            # and concatenates their complete token sequences (left, right).
            extractor = ConvChannelFeatureExtractor(
                **extractor_kwargs, share_weights_over_channels=False
            )
        else:
            extractor = ConvFeatureExtractor(**extractor_kwargs)
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
            use_gradient_checkpointing=cfg["use_gradient_checkpointing"],
            compact_masked_tokens=cfg["compact_masked_tokens"],
            position_channels=self.input_channels,
        )
        self.masker = TimeInverseBlockMasker(
            **cfg["masker"], channel_based_masking=self.input_channels == 2
        )
        if checkpoint_path is not None:
            self.load_official_checkpoint(checkpoint_path)
        elif pretrained_path is not None:
            self.load_huggingface_checkpoint(pretrained_path)
        elif pretrained:
            self.load_huggingface_checkpoint(
                download_official_weights(
                    local_files_only=local_files_only,
                    variant="nat" if self.input_channels == 2 else "base",
                )
            )

    @classmethod
    def from_pretrained(
        cls, *, local_files_only: bool = False, variant: str = "base"
    ) -> "WavJEPABackbone":
        """Load the pinned official mono or Nat release without trust_remote_code."""
        if variant not in ("base", "nat"):
            raise ValueError("variant must be base or nat")
        return cls(
            {"input_channels": 2 if variant == "nat" else 1},
            pretrained=True,
            local_files_only=local_files_only,
        )

    @staticmethod
    def _validate_config(cfg: dict[str, Any]) -> None:
        if type(cfg["input_channels"]) is not int or cfg["input_channels"] not in (1, 2):
            raise ValueError("input_channels must be 1 (mono) or 2 (binaural)")
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
            if cfg["input_channels"] == 2 and width % 4:
                raise ValueError(f"binaural {prefix}_dim must be divisible by four")
        if not 1 <= cfg["average_top_k_layers"] <= cfg["encoder_layers"]:
            raise ValueError("average_top_k_layers must fit the encoder depth")
        if cfg["compile_modules"] is not False:
            raise ValueError("compile_modules is unsupported")
        if type(cfg["use_gradient_checkpointing"]) is not bool:
            raise ValueError("use_gradient_checkpointing must be boolean")
        if type(cfg["compact_masked_tokens"]) is not bool:
            raise ValueError("compact_masked_tokens must be boolean")
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

    def _normalized_windows(
        self, waveforms: Tensor, *, window_samples: int | None = None
    ) -> Tensor:
        window_samples = self.window_samples if window_samples is None else window_samples
        expected_ndim = 3 if self.input_channels == 1 else 4
        if (
            waveforms.ndim != expected_ndim
            or waveforms.shape[-1] != window_samples
            or (self.input_channels == 2 and waveforms.shape[-2] != 2)
        ):
            shape = (
                f"[B,T,{window_samples}]"
                if self.input_channels == 1
                else f"[B,T,2,{window_samples}]"
            )
            raise ValueError(f"expected waveform {shape}")
        if min(waveforms.shape) < 1 or not waveforms.is_floating_point():
            raise ValueError("waveforms must be nonempty floating-point tensors")
        if not torch.isfinite(waveforms).all():
            raise ValueError("waveforms must contain only finite values")
        flat = waveforms.reshape(-1, self.input_channels, window_samples).float()
        # Exactly the original per-crop mean/std transform, without random crops,
        # shuffling or forced bfloat16. Binaural normalization uses both ears
        # jointly so relative levels are retained. Precision belongs to the trainer.
        return (flat - flat.mean(dim=(-2, -1), keepdim=True)) / (
            flat.std(dim=(-2, -1), keepdim=True) + 1e-5
        )

    def encode_tokens(self, waveforms: Tensor, teacher: bool = False) -> Tensor:
        """Return [batch, windows, tokens, width] with fixed positional embeddings.

        Binaural tokens are channel-major: all left-ear tokens, then all right-ear
        tokens; positions encode both time and channel. Windows pool both ears.

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

    def _encode_tokens(
        self, waveforms: Tensor, teacher: bool, readout_mode: str = "legacy"
    ) -> Tensor:
        x = self._normalized_windows(waveforms)
        x = self.model.feature_norms(self.model.extract_audio(x))
        if self.model.post_extraction_mapper is not None:
            x = self.model.post_extraction_mapper(x)
        x = x + self.model.pos_encoding_encoder
        if teacher:
            x = self.model._forward_teacher(x.detach())
        elif readout_mode == "normalized_top_k":
            x = self.model.encoder_top_k_forward(x)
        else:
            x = self.model.encoder_forward(x)
        return x.reshape(
            waveforms.shape[0], waveforms.shape[1], self.model.total_patches, self.embedding_dim
        )

    def token_timestamps_ms(self) -> Tensor:
        """Nominal centers of convolutional support, relative to window start."""
        samples = (
            torch.arange(self.model.total_patches // self.input_channels, dtype=torch.float64)
            * self.stride_samples
            + (self.receptive_field_samples - 1) / 2
        )
        return samples.repeat(self.input_channels) * (1000.0 / self.sample_rate)

    def encode_windows(self, waveforms: Tensor, readout_mode: str = "legacy") -> Tensor:
        """Read completed windows with gradients, using an explicit saved convention.

        ``legacy`` preserves the final-layer readout of existing checkpoints.
        ``normalized_top_k`` applies the teacher's exact layer-selection and
        normalization rule to student layers, before the same temporal mean.
        The latter aligns the distributions presented to online and EMA upper
        maps; it does not alter the original masked prediction objective.
        """
        if readout_mode not in {"legacy", "normalized_top_k"}:
            raise ValueError("readout_mode must be legacy or normalized_top_k")
        return (
            self._encode_tokens(waveforms, teacher=False, readout_mode=readout_mode)
            .float()
            .mean(dim=2)
        )

    def masked_loss(self, waveforms: Tensor, seed: int | None = None) -> Tensor:
        """Original masked objective on independent crops, permitting another length.

        Nat can use eight random two-second crops while its temporal hierarchy
        reads shorter chronological windows. The learned modules are shared;
        only fixed positions and sampled mask lengths vary. No model parameter
        or fixed-window readout configuration is changed by this call.
        """
        if waveforms.ndim not in (3, 4):
            raise ValueError("masked crops require [B,T,S] or [B,T,2,S] waveforms")
        return self._masked_loss_with_targets(
            waveforms, seed=seed, window_samples=waveforms.shape[-1]
        )[0]

    def masked_loss_with_targets(
        self, waveforms: Tensor, seed: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return masked loss and detached [B,T,D] teacher means from one forward.

        Reusing these targets avoids repeating the complete teacher encoder for
        the forecasting objective. The mask affects only student/decoder paths.
        """
        return self._masked_loss_with_targets(
            waveforms, seed=seed, window_samples=self.window_samples
        )

    def _masked_loss_with_targets(
        self, waveforms: Tensor, *, seed: int | None, window_samples: int
    ) -> tuple[Tensor, Tensor]:
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise ValueError("seed must be an integer in [0,2**63) or None")
        if seed is None:
            # Follow Lightning/PyTorch's seeded training stream rather than
            # NumPy default_rng(None), which would draw untracked OS entropy.
            seed = int(torch.randint(0, 2**63 - 1, (), device="cpu").item())
        audio = self._normalized_windows(waveforms, window_samples=window_samples)
        tokens_per_channel = window_samples
        for _, kernel, stride in self.config["conv_layers"]:
            tokens_per_channel = (tokens_per_channel - kernel) // stride + 1
        mask_config = self.config["masker"]
        for probability, length in (
            ("context_mask_prob", "context_mask_length"),
            ("target_prob", "target_length"),
        ):
            if (
                mask_config[length] >= tokens_per_channel
                or mask_config[probability] * tokens_per_channel / mask_config[length] < 1
            ):
                raise ValueError("masked crop is too short for the configured frontend and masks")
        token_count = tokens_per_channel * self.input_channels
        masks = self.masker(
            audio.shape[0],
            token_count,
            self.input_channels,
            generator=np.random.default_rng(seed),
        )
        # The official masker constructs CPU tensors. Check them before the
        # transfer so validity checks do not introduce CUDA-to-host barriers.
        if not masks[1].any() or not (~masks[0]).any(dim=1).all():
            raise RuntimeError("upstream masker produced an empty target or context")
        masks = tuple(mask.to(audio.device) for mask in masks)
        positions = None
        if token_count != self.model.total_patches:

            def position_tensor(width: int) -> Tensor:
                if self.input_channels == 2:
                    values = get_binaural_sincos_pos_embed(width, tokens_per_channel)
                else:
                    values = get_1d_sincos_pos_embed_from_grid(
                        width, np.arange(tokens_per_channel, dtype=np.float64)
                    )
                return torch.from_numpy(values).float().unsqueeze(0).to(audio.device)

            positions = (
                position_tensor(self.config["encoder_dim"]),
                position_tensor(self.config["decoder_dim"]),
            )
        output = self.model(audio, *masks, position_embeddings=positions)
        targets = output["targets"].detach().float().mean(dim=1)
        targets = targets.reshape(waveforms.shape[0], waveforms.shape[1], self.embedding_dim)
        return output["loss"], targets

    @torch.no_grad()
    def encode_teacher_windows(self, waveforms: Tensor) -> Tensor:
        """Read out upstream normalized top-layer targets, without cross-window context.

        As in WavJEPA, only the context Transformer is EMA: frontend, feature norm
        and mapper are shared with the student and detached on this target path.
        """
        return self.encode_tokens(waveforms, teacher=True).float().mean(dim=2)

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

        Only fixed sinusoidal positions may be regenerated when adapting the
        official 200-token-per-ear base/Nat release to another window length.
        No learned-tensor resizing, missing-key fallback or remote Python is allowed. Local files may be used offline;
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
        normalized = self._adapt_fixed_positions(normalized)
        self._load_checked_state(normalized)
        self.checkpoint_sha256 = file_sha256(source)

    def _adapt_fixed_positions(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Adapt only validated, nonlearned base-release sinusoidal positions."""
        expected = self.model.state_dict()
        position_keys = ("pos_encoding_encoder", "pos_encoding_decoder")
        if not any(
            key in state and state[key].shape != expected[key].shape for key in position_keys
        ):
            return state
        if any(
            self.config[key] != DEFAULT_CONFIG[key]
            for key in DEFAULT_CONFIG
            if key
            not in {
                "masker",
                "input_channels",
                "window_samples",
                "use_gradient_checkpointing",
                "compact_masked_tokens",
            }
        ):
            raise ValueError(
                "positional adaptation requires the official base learned architecture"
            )
        adapted = dict(state)
        for key, width in zip(position_keys, (768, 384), strict=True):
            original = state.get(key)
            if self.input_channels == 2:
                canonical_array = get_binaural_sincos_pos_embed(width, 200)
            else:
                canonical_array = get_1d_sincos_pos_embed_from_grid(
                    width, np.arange(200, dtype=np.float64)
                )
            canonical = torch.from_numpy(canonical_array).float().unsqueeze(0)
            if (
                original is None
                or original.dtype != torch.float32
                or not torch.equal(original, canonical)
            ):
                raise ValueError(
                    "positional adaptation requires exact official 200-token-per-ear sinusoidal tensors"
                )
            adapted[key] = expected[key].clone()
        return adapted

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

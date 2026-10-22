"""Explicit safe inference loader for upstream-backed dynamics checkpoints.

The legacy src.inference loader remains a different model family; src.embed
selects this loader by default and the legacy loader with --model-family legacy.
No checkpoint-controlled Hydra target or arbitrary serialized object is executed.
"""

from functools import partial
from pathlib import Path

import torch

from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule


def load_model(path: str | Path, device: str = "cpu") -> HierarchicalWavJEPAModule:
    """Load a trusted local new-family checkpoint with its exact semantic contract."""
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("checkpoint must be a trusted local file")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("expected a hierarchical dynamics checkpoint dictionary")
    metadata = HierarchicalWavJEPAModule.validated_checkpoint_metadata(checkpoint)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("checkpoint must contain a tensor-only state_dict")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains nonfinite state")
    # Constructors are explicit local code, never checkpoint-controlled imports.
    # Reconstruction initializes weights before loading; do not advance the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        model = HierarchicalWavJEPAModule(
            backbone=WavJEPABackbone(config=metadata["backbone"]),
            optimizer=partial(torch.optim.AdamW, lr=1e-4),
            **metadata["dynamics"],
        )
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)

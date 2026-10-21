"""Load a trusted H-WavJEPA checkpoint without executing a saved Hydra config."""

from functools import partial
from pathlib import Path

import torch

from src.models.components.hierarchy import HierarchyEncoder
from src.models.hwavjepa_module import HWavJEPAModule


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> HWavJEPAModule:
    """Restore architecture and weights from this project's versioned checkpoint.

    Only trusted local checkpoints are supported. PyTorch's weights-only loader
    limits pickle execution but does not make hostile files safe or bound memory.
    No optimizer state is applied and no config-specified Python target executes.
    """
    path = Path(checkpoint).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("checkpoint must be a trusted local file")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a Lightning state dictionary")
    metadata = HWavJEPAModule.validated_checkpoint_metadata(state)
    if not isinstance(state.get("state_dict"), dict):
        raise ValueError("checkpoint is missing its model state_dict")
    # Constructors initialize weights before replacement; preserve callers' RNG.
    with torch.random.fork_rng(devices=[]):
        model = HWavJEPAModule(
            encoder=HierarchyEncoder(**metadata["encoder"]),
            optimizer=partial(torch.optim.AdamW),
            **metadata["module"],
        )
    model.on_load_checkpoint(state)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)

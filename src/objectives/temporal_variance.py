"""Within-recording variance pressure for action-free upper audio spaces.

This is an explicit audio adaptation, separate from the cross-recording SIGReg
objective. A positive penalty for recording-constant states does not prove that
the changing information is semantic or guarantee escape from exact collapse.
"""

import math

import torch
from torch import Tensor


def temporal_variance_floor(
    states: Tensor, *, std_floor: float = 0.1, epsilon: float = 1e-4
) -> Tensor:
    """Average positive std shortfall independently within each recording.

    Input is [recording, valid dense time, feature]. Time is never mixed with
    recordings: diverse recording identities cannot satisfy this objective.
    Compute in float32 under mixed precision. Equal-size DDP local batch means
    already produce the global recording mean under standard gradient averaging.
    The epsilon stabilizes gradients but makes the exact constant state stationary.
    """
    if states.ndim != 3 or states.shape[0] < 1 or states.shape[1] < 2 or states.shape[2] < 1:
        raise ValueError("temporal variance needs nonempty [batch, time >= 2, feature] states")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("temporal variance epsilon must be finite and positive")
    if not math.isfinite(std_floor) or std_floor <= math.sqrt(epsilon):
        raise ValueError("temporal std floor must be finite and exceed sqrt(epsilon)")
    std = torch.sqrt(states.float().var(dim=1, unbiased=False) + epsilon)
    return torch.relu(std_floor - std).mean()

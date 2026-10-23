"""Held-out checks for recording-constant representations, not training penalties."""

import torch
from torch import Tensor


@torch.no_grad()
def recording_variance(states: Tensor) -> dict[str, Tensor]:
    """Decompose a [recording, time, feature] population's variance.

    Both terms use population (not unbiased sample) variance. With equal-length
    trajectories, within + between equals the population's total variance.
    Aligned distributed validation gathers the global recording batch first;
    standalone calls describe the supplied local population. Averaging these
    batch estimates is not whole-dataset variance.
    A recording-constant solution has zero within variance even if its batch
    distribution is diverse. A fully collapsed solution makes both terms zero.
    """
    if states.ndim != 3 or min(states.shape) < 1:
        raise ValueError("states must be nonempty [recording, time, feature]")
    states = states.detach().float()
    within = states.var(dim=1, unbiased=False).mean()
    between = states.mean(dim=1).var(dim=0, unbiased=False).mean()
    total = within + between
    return {
        "within_recording_variance": within,
        "between_recording_variance": between,
        # Zero for an exactly collapsed population; inspect the absolute terms
        # alongside this ratio rather than treating zero as a sufficient verdict.
        "within_variance_fraction": torch.where(total > 0, within / total.clamp_min(1e-30), 0),
    }


@torch.no_grad()
def forecast_diagnostics(
    prediction: Tensor,
    target: Tensor,
    teacher_origin: Tensor,
    shifted_context_prediction: Tensor,
) -> dict[str, Tensor]:
    """Compare forecasts with teacher-space persistence and time-shift controls.

    Persistence repeats the EMA state of the final observed window, so it does
    not confuse student/teacher drift with temporal change. The shifted forecast
    uses a one-position cyclic shift of context states while keeping their
    position labels fixed. This is a deterministic order-sensitivity probe, not
    an unbiased random-shuffle estimate or evidence of semantic abstraction.
    """
    forecast = (prediction.float() - target.float()).square().mean()
    persistence = (teacher_origin.float() - target.float()).square().mean()
    shifted = (shifted_context_prediction.float() - target.float()).square().mean()
    return {
        "persistence_mse": persistence,
        "forecast_gain_over_persistence": persistence - forecast,
        "time_shifted_context_mse": shifted,
        "time_shifted_context_mse_increase": shifted - forecast,
    }

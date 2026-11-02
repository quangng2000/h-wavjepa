"""Learned causal aggregation on dense completed-window embeddings."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def validate_aggregation_mode(mode: str, hierarchy: bool) -> None:
    """Keep learned aggregation out of the parameter-free flat control."""
    if mode not in ("mean", "conv1d"):
        raise ValueError("aggregation_mode must be mean or conv1d")
    if mode == "conv1d" and not hierarchy:
        raise ValueError("conv1d aggregation requires hierarchy=true")


class CausalConv1d(nn.Module):
    """Full channel-mixing Conv1D, aligned to the last input in each window.

    Inputs and outputs are [batch, time, feature]. Only complete windows are
    returned. Dilation follows the preceding level's cumulative stride; the
    caller subsequently selects physical endpoints on the output stride grid.
    The initial diagonal averaging kernel preserves the mean arm's features.
    """

    def __init__(self, width: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (width, kernel_size, dilation)):
            raise ValueError("width, kernel_size and dilation must be positive integers")
        self.history = (kernel_size - 1) * dilation
        # Do not change initialization of the MLPs, predictors or backbone in
        # a matched seeded comparison just because the Conv arm adds weights.
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv1d(width, width, kernel_size, dilation=dilation, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            index = torch.arange(width)
            self.conv.weight[index, index, :] = 1 / kernel_size

    def forward(self, states: Tensor) -> Tensor:
        if states.ndim != 3 or states.shape[-1] != self.conv.in_channels:
            raise ValueError(
                "states must have shape [batch, time, feature] with the configured width"
            )
        if states.shape[1] <= self.history:
            return states[:, :0]
        return self.conv(states.transpose(1, 2)).transpose(1, 2)


def with_convolutions(
    abstractions: nn.ModuleList, width: int, geometry: tuple[Sequence[int], Sequence[int]]
) -> nn.ModuleList:
    """Include online Conv parameters in the existing abstraction/EMA contract."""
    windows, strides = geometry
    return nn.ModuleList(
        nn.Sequential(CausalConv1d(width, window, stride), abstract)
        for abstract, window, stride in zip(abstractions, windows[1:], strides[:-1], strict=True)
    )

"""Tiny noncausal masked predictors, not the released WavJEPA architecture."""

import math

import torch
from torch import Tensor, nn


class MaskedPredictor(nn.Module):
    """Bidirectional context attention with positional masked-token queries."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        heads = next(head for head in (4, 2, 1) if width % head == 0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Linear(width, width)

    def forward(self, features: Tensor, hidden: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.width:
            raise ValueError("features must have shape [B,N,width]")
        if hidden.shape != features.shape[:2] or hidden.dtype != torch.bool:
            raise ValueError("hidden must be boolean [B,N]")
        values = torch.where(hidden.unsqueeze(-1), self.mask_token.to(features.dtype), features)
        positions = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, device=values.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.width)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.zeros(values.shape[1], self.width, device=values.device)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : self.width // 2])
        return self.output(self.context(values + encoding.to(values.dtype).unsqueeze(0)))

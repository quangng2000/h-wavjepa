"""Small causal forecasting head built from PyTorch Transformer modules."""

import torch
from torch import Tensor, nn


class TemporalPredictor(nn.Module):
    def __init__(self, width: int, context_windows: int, layers: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("predictor width must be divisible by predictor_heads")
        self.position = nn.Embedding(context_windows, width)
        layer_options = dict(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options),
            layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones one layer with identical initial values.
        # Construct subsequent layers independently using PyTorch's initializer,
        # as recommended by the TransformerEncoder documentation.
        for index in range(1, layers):
            self.transformer.layers[index] = nn.TransformerEncoderLayer(**layer_options)
        self.output = nn.Linear(width, width)

    def forward(self, states: Tensor, positions: Tensor) -> Tensor:
        # Absolute window indices preserve the physical spacing of subsampled
        # states: [0,2,4] is not relabeled as [0,1,2]. All inputs end at the same
        # prediction origin, regardless of the level's stride or future horizon.
        values = states + self.position(positions).unsqueeze(0)
        mask = torch.ones(
            len(positions), len(positions), dtype=torch.bool, device=states.device
        ).triu(diagonal=1)
        return self.output(self.transformer(values, mask=mask)[:, -1])

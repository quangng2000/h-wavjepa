"""SIGReg from H-JEPA, MIT licensed; see third_party/licenses/H-JEPA.txt.

Source: kevinghst/H-JEPA, commit 97054ecaff83e13839195ec4b92d74c08a0127c5,
h_jepa/loss.py, SIGReg. The single-device mathematical implementation is
preserved. Distributed execution is explicitly rejected instead of inheriting
upstream's unvalidated distributed reduction. Input is [time, recordings, width],
not a flattened mixture of time and recordings. This is a collapse penalty,
not a guarantee of semantic information or optimizer convergence.
"""

import torch
from torch import nn

SOURCE_COMMIT = "97054ecaff83e13839195ec4b92d74c08a0127c5"


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 128) -> None:
        super().__init__()
        if type(knots) is not int or knots < 2 or type(num_proj) is not int or num_proj < 1:
            raise ValueError("SIGReg needs knots >= 2 and num_proj >= 1")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3 or proj.size(1) < 2 or not torch.isfinite(proj).all():
            raise ValueError("SIGReg requires finite [time, batch >= 2, width] states")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError("Distributed SIGReg is not validated in this integration")
        if proj.dtype != torch.float32:
            raise ValueError("SIGReg integration currently requires float32")
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

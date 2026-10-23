"""LeJEPA's SIGReg, using H-JEPA code under MIT; see third_party/licenses/H-JEPA.txt.

Method: Balestriero and LeCun, LeJEPA, https://arxiv.org/abs/2511.08544.

Source: kevinghst/H-JEPA, commit 97054ecaff83e13839195ec4b92d74c08a0127c5,
h_jepa/loss.py, SIGReg. The single-device mathematical implementation is
preserved. Distributed training gathers the actual cross-rank population with
autograd; statistics stay in FP32 under autocast. Input is [time, recordings, width],
not a flattened mixture of time and recordings. This is a collapse penalty,
not a guarantee of semantic information or optimizer convergence.
"""

import torch
from torch import nn

from src.objectives.distributed import prepare_population, regularizer_rng

SOURCE_COMMIT = "97054ecaff83e13839195ec4b92d74c08a0127c5"


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 128) -> None:
        super().__init__()
        if type(knots) is not int or knots < 2 or type(num_proj) is not int or num_proj < 1:
            raise ValueError("SIGReg needs knots >= 2 and num_proj >= 1")
        self.num_proj = num_proj
        self._distributed_rng = torch.Generator().manual_seed(torch.initial_seed())
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(
        self, proj: torch.Tensor, *, global_batch: bool | None = None, seed: int | None = None
    ) -> torch.Tensor:
        if global_batch is None:
            global_batch = self.training and torch.is_grad_enabled()
        proj, distributed = prepare_population(
            proj, global_batch=global_batch, seed=seed, name="SIGReg"
        )
        with (
            torch.autocast(device_type=proj.device.type, enabled=False),
            regularizer_rng(
                proj, distributed=distributed, seed=seed, generator=self._distributed_rng
            ),
        ):
            a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
            a = a.div_(a.norm(p=2, dim=0))
            x_t = (proj @ a).unsqueeze(-1) * self.t
            cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
            err = (cos - self.phi).square() + sin.square()
            statistic = (err @ self.weights) * proj.size(-2)
            return statistic.mean()

"""Global-recording populations for distribution regularizers.

Every rank evaluates the same loss on an autograd-aware all-gather. Its backward
sums the contributions from every rank; DDP's parameter-gradient averaging then
produces exactly the gradient of one loss over the global population. Do not
divide the loss by world size. Other local-mean objectives require equal local
batch sizes, which this adapter checks collectively before gathering activations.

Evaluation defaults to local statistics and performs no collectives: validation
loaders may have unequal lengths. Gradient accumulation does not enlarge the
population used by an individual invocation.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def prepare_population(
    states: torch.Tensor, *, global_batch: bool, seed: int | None, name: str
) -> tuple[torch.Tensor, bool]:
    """Validate every rank before collecting equal-sized FP32 populations."""
    distributed = global_batch and dist.is_available() and dist.is_initialized()
    supported = states.dtype in {torch.float32, torch.bfloat16}
    dimensions = states.ndim == 3 and all(size > 0 for size in states.shape)
    seed_valid = seed is None or (type(seed) is int and 0 <= seed < 2**63)
    valid = (
        dimensions
        and supported
        and states.device.type in {"cpu", "cuda"}
        and seed_valid
        and bool(torch.isfinite(states).all())
    )
    message = (
        f"{name} requires finite CPU/CUDA float32 or bfloat16 "
        "[time >= 1, batch >= 2 globally, width >= 1] states and a valid seed"
    )
    if distributed:
        # Fixed-size metadata lets every rank reject invalid/mismatched inputs
        # together instead of leaving peers blocked in the activation collective.
        shape = list(states.shape) if states.ndim == 3 else [-1, -1, -1]
        metadata = torch.tensor(
            [int(valid), *shape, int(states.dtype == torch.bfloat16)],
            device=states.device,
            dtype=torch.int64,
        )
        gathered = [torch.empty_like(metadata) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, metadata)
        descriptions = torch.stack(gathered).cpu().tolist()
        if any(not item[0] for item in descriptions):
            raise ValueError(f"{message}; invalid input on at least one distributed rank")
        if any(item[1:] != descriptions[0][1:] for item in descriptions[1:]):
            raise ValueError(f"{name} requires equal local shapes and dtypes on every rank")
        if shape[1] * dist.get_world_size() < 2:
            raise ValueError(message)
        # Floating cast retains the local graph, including BF16 producer gradients.
        states = torch.cat(all_gather(states.float().contiguous()), dim=1)
    else:
        if not valid or states.size(1) < 2:
            raise ValueError(message)
        states = states.float()
    return states, distributed


@contextmanager
def regularizer_rng(
    states: torch.Tensor,
    *,
    distributed: bool,
    seed: int | None,
    generator: torch.Generator,
) -> Iterator[None]:
    """Use identical random sketches/cluster initialization without global RNG changes.

    Explicit seeds make a distributed training step reproducible across checkpoint
    resumes. Without one, a private per-regularizer stream supplies rank-zero seeds.
    Single-device calls with no seed preserve the upstream RNG behavior exactly.
    """
    if distributed:
        if dist.get_rank() == 0 and seed is None:
            seed = int(torch.randint(2**63 - 1, (), generator=generator))
        shared_seed = torch.tensor(seed or 0, dtype=torch.int64, device=states.device)
        dist.broadcast(shared_seed, src=0)
        seed = int(shared_seed.item())
    devices = [states.device.index or 0] if states.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
        yield

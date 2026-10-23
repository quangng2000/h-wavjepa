"""Durable checkpoint publication hands the same outcome to both real DDP ranks."""

import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightning.pytorch.strategies import DDPStrategy

from scripts.segment_checkpoint import DurableStepCheckpoint
from src.utils.runtime import file_sha256


class DelayedPublication(DurableStepCheckpoint):
    """Replace only remote storage; retain the actual local save and broadcasts."""

    def __init__(self, *, fail, **kwargs):
        super().__init__(**kwargs)
        self.fail = fail
        self.publication_calls = 0

    def _publish(self, path, step):
        assert dist.get_rank() == 0
        assert torch.load(path, weights_only=True) == {"global_step": step}
        self.publication_calls += 1
        # Peers must wait for publication before accepting its receipt or error.
        time.sleep(0.2)
        if self.fail:
            raise TimeoutError("private-storage-url-with-secret-fixture")
        receipt = {
            "checkpoint": {"step": step, "sha256": file_sha256(path), "bytes": path.stat().st_size},
            "verified": True,
        }
        self.receipt_path.write_text(json.dumps(receipt))
        return receipt


def _checkpoint_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=121)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timeout
    )
    try:
        root = Path(directory)
        strategy = DDPStrategy(process_group_backend="gloo", timeout=timeout)

        def save_checkpoint(path, weights_only):
            assert weights_only is False
            if rank == 0:
                torch.save({"global_step": 1}, path)
            strategy.barrier()

        trainer = SimpleNamespace(
            is_global_zero=rank == 0,
            global_step=1,
            strategy=strategy,
            save_checkpoint=save_checkpoint,
            loggers=[],
        )
        results = []
        for phase, fail in (("success", False), ("timeout", True)):
            callback = DelayedPublication(
                fail=fail,
                repository_options={},
                segment_id=f"segment-{phase}",
                data_seed=42,
                receipt_path=str(root / f"{phase}-receipt.json"),
                transfer_timeout_seconds=1,
                hard_deadline=time.time() + 60,
                dirpath=root,
                every_n_train_steps=1,
            )
            error = None
            try:
                callback._save_checkpoint(trainer, str(root / f"{phase}.ckpt"))
            except RuntimeError as exception:
                error = str(exception)
            results.append(
                {
                    "phase": phase,
                    "receipt": callback.last_receipt,
                    "error": error,
                    "publication_calls": callback.publication_calls,
                }
            )
        (root / f"rank-{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo backend")
def test_delayed_checkpoint_publication_broadcasts_success_and_timeout_to_both_ranks(tmp_path):
    context = mp.spawn(
        _checkpoint_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Checkpoint handoff exceeded the bounded CPU/Gloo test deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    checkpoint = tmp_path / "success.ckpt"
    expected_receipt = {
        "checkpoint": {
            "step": 1,
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "verified": True,
    }
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    for rank, (success, failure) in enumerate(reports):
        assert success["receipt"] == expected_receipt
        assert success["error"] is None
        assert failure["receipt"] is None
        assert failure["error"] == "Durable checkpoint publication failed (TimeoutError); stopping"
        assert success["publication_calls"] == failure["publication_calls"] == int(rank == 0)
    assert json.loads((tmp_path / "success-receipt.json").read_text()) == expected_receipt
    assert not (tmp_path / "timeout-receipt.json").exists()

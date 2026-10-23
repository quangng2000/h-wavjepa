"""Operator stop requests follow the ordinary full-checkpoint lifecycle."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import lightning as L
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from lightning.pytorch.callbacks import Callback
from test_step_checkpoint import InfiniteOnes, StepCheckpoint, TinyModel, trainer_options
from torch.utils.data import DataLoader

from scripts.operator_stop import MAX_REQUEST_BYTES, OperatorStop
from src.utils.runtime import atomic_json


def callback(tmp_path):
    return OperatorStop(str(tmp_path / "stop.json"), "training-run", "segment-005", "a" * 32)


@pytest.mark.parametrize("rank_zero", [True, False])
def test_only_rank_zero_reads_and_all_ranks_share_stop_decision(tmp_path, rank_zero):
    cb = callback(tmp_path)
    cb._is_requested = Mock(return_value=True)
    broadcast = Mock(return_value=True)
    trainer = SimpleNamespace(
        global_step=500,
        is_global_zero=rank_zero,
        should_stop=False,
        strategy=SimpleNamespace(broadcast=broadcast),
    )
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop
    assert cb.stop_requested
    assert cb._is_requested.call_count == int(rank_zero)
    broadcast.assert_called_once_with(rank_zero, src=0)
    cb.on_train_batch_end(trainer, None, None, None, 1)
    assert broadcast.call_count == 1


@pytest.mark.parametrize("mutation", ["run_id", "segment_id", "nonce", "schema_version", "extra"])
def test_stale_or_unbound_request_is_ignored(tmp_path, mutation):
    cb = callback(tmp_path)
    request = dict(cb.expected)
    request[mutation] = True if mutation == "schema_version" else "wrong"
    atomic_json(cb.request_path, request)
    assert cb._is_requested() is False
    atomic_json(cb.request_path, cb.expected)
    assert cb._is_requested() is True


def test_missing_malformed_oversized_symlink_and_fifo_requests_are_safe(tmp_path, capsys):
    cb = callback(tmp_path)
    assert cb._is_requested() is False
    cb.request_path.write_bytes(b"{private-malformed")
    assert cb._is_requested() is False
    assert cb._is_requested() is False
    assert capsys.readouterr().out.count("operator_stop_request_rejected") == 1
    cb.request_path.write_bytes(b"x" * (MAX_REQUEST_BYTES + 1))
    assert cb._is_requested() is False
    cb.request_path.write_bytes(b"[" * 1500 + b"]" * 1500)
    assert cb._is_requested() is False
    cb.request_path.unlink()
    target = tmp_path / "target.json"
    atomic_json(target, cb.expected)
    cb.request_path.symlink_to(target)
    assert cb._is_requested() is False
    cb.request_path.unlink()
    import os

    os.mkfifo(cb.request_path)
    assert cb._is_requested() is False


def test_no_request_never_clears_an_existing_stop(tmp_path):
    cb = callback(tmp_path)
    trainer = SimpleNamespace(
        global_step=1,
        is_global_zero=True,
        should_stop=True,
        strategy=SimpleNamespace(broadcast=lambda value, src: value),
    )
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert trainer.should_stop


def _stop_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30)
    )
    try:
        from pathlib import Path

        cb = callback(Path(directory))
        if rank:
            cb._is_requested = Mock(side_effect=AssertionError("Peer read operator file"))

        def broadcast(value, src):
            values = [value]
            dist.broadcast_object_list(values, src=src)
            return values[0]

        trainer = SimpleNamespace(
            global_step=1,
            is_global_zero=rank == 0,
            should_stop=False,
            strategy=SimpleNamespace(broadcast=broadcast),
        )
        cb.on_train_batch_end(trainer, None, None, None, 0)
        assert not trainer.should_stop
        if rank == 0:
            atomic_json(cb.request_path, cb.expected)
        dist.barrier()
        trainer.global_step = 2
        cb.on_train_batch_end(trainer, None, None, None, 1)
        assert trainer.should_stop
        assert cb.stop_requested
        if rank:
            cb._is_requested.assert_not_called()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo")
def test_real_two_rank_operator_stop_agreement(tmp_path):
    mp.spawn(
        _stop_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True
    )


@pytest.mark.integration
def test_real_lightning_operator_stop_saves_full_state_and_resumes(tmp_path):
    stop = callback(tmp_path)

    class RequestAtTwo(Callback):
        def on_train_batch_end(self, trainer, *args):
            if trainer.global_step == 2:
                atomic_json(stop.request_path, stop.expected)

    saved = StepCheckpoint(dirpath=tmp_path / "checkpoints", every_n_train_steps=100)
    # Lightning places checkpoint callbacks last even if supplied first.
    trainer = L.Trainer(
        max_steps=10,
        callbacks=[saved, RequestAtTwo(), stop],
        limit_val_batches=0,
        **trainer_options(tmp_path),
    )
    trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert trainer.global_step == 2
    state = torch.load(saved.last_model_path, map_location="cpu", weights_only=True)
    assert state["global_step"] == 2
    assert state["optimizer_states"][0]["state"]
    # The next launch has a different nonce: the old marker cannot stop it.
    replacement = OperatorStop(str(stop.request_path), "training-run", "segment-005", "b" * 32)
    resumed = L.Trainer(
        max_steps=3,
        callbacks=[replacement],
        enable_checkpointing=False,
        limit_val_batches=0,
        **trainer_options(tmp_path / "resumed"),
    )
    resumed.fit(
        TinyModel(),
        ckpt_path=saved.last_model_path,
        train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2),
    )
    assert resumed.global_step == 3

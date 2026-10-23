"""Background publication preserves immutable full state and DDP ordering."""

import json
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from test_multihost_segments import config_plan
from test_step_checkpoint import InfiniteOnes, TinyModel, trainer_options
from test_training_segments import CheckpointS3
from torch.utils.data import DataLoader

from scripts.segment_checkpoint import CheckpointRepository, DurableStepCheckpoint
from scripts.train_segment import make_config


def callback_options(directory):
    return dict(
        repository_options={},
        segment_id="segment-async",
        data_seed=42,
        receipt_path=str(directory / "receipt.json"),
        transfer_timeout_seconds=1,
        hard_deadline=time.time() + 60,
        dirpath=directory / "recovery",
        filename="step_{step:07d}",
        auto_insert_metric_name=False,
        every_n_train_steps=1,
        async_upload=True,
    )


def receipt(step, path):
    return {"checkpoint": {"step": step, "sha256": "a" * 64, "bytes": path.stat().st_size}}


@pytest.mark.parametrize("lineage", ["batch_transition", "curriculum_source"])
def test_hydra_nested_lineage_publishes_actual_checkpoint_and_verified_receipt(
    tmp_path, monkeypatch, lineage
):
    client = CheckpointS3()
    options = {
        "bucket": "checkpoint-only",
        "prefix": "training/target-run-001",
        "run_id": "target-run-001",
        "index_sha256": "a" * 64,
        "base_data_seed": 42,
        "global_batch": 144,
        # Only the in-memory test double bypasses the SDK upload adapter.
        "managed_upload": False,
    }
    if lineage == "batch_transition":
        options.update(
            clean_binaural=True,
            batch_transition={
                "source_run_id": "source-run-001",
                "source_prefix": "training/source-run-001",
                "source_checkpoint_sha256": "b" * 64,
                "source_optimizer_step": 500,
                "old_global_batch": 16,
                "new_global_batch": 144,
            },
        )
    else:
        options.update(
            naturalistic_manifest_sha256="c" * 64,
            curriculum_source={
                "run_id": "source-run-001",
                "prefix": "training/source-run-001",
                "checkpoint_sha256": "b" * 64,
            },
        )
    repository = CheckpointRepository(client, **options, check=lambda: None)
    repository.open(initialize=True)
    endpoint = "https://storage.example.invalid"
    if lineage == "batch_transition":
        cfg = make_config(
            config_plan(tmp_path)
            | {
                "distributed": {"nnodes": 1, "nproc_per_node": 8},
                "global_batch": 144,
                "resume_step": 500,
                "resolved_data_seed": 542,
                "repository_options": options | {"endpoint": endpoint},
            }
        ).callbacks.model_checkpoint
    else:
        cfg = OmegaConf.create(
            {"_target_": "scripts.segment_checkpoint.DurableStepCheckpoint"}
            | callback_options(tmp_path)
            | {"repository_options": options | {"endpoint": endpoint}, "data_seed": 542}
        )
    assert isinstance(cfg.repository_options[lineage], DictConfig)
    callback = instantiate(cfg)
    assert type(callback.repository_options[lineage]) is dict
    assert callback.repository_options[lineage] == options[lineage]
    monkeypatch.setattr("scripts.segment_checkpoint.checkpoint_client", lambda _: client)

    def save_checkpoint(path, weights_only):
        assert weights_only is False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"global_step": 750, "state_dict": {"weight": torch.ones(2)}}, path)

    trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=750,
        strategy=SimpleNamespace(broadcast=lambda value: value),
        save_checkpoint=save_checkpoint,
        loggers=[],
    )
    path = tmp_path / "recovery" / "step_0000750.ckpt"
    try:
        # Exercise the real background publisher, constructor, SHA readback,
        # receipt serialization and callback receipt propagation after Hydra.
        callback._save_checkpoint(trainer, str(path))
        callback._drain_publication(trainer)
    finally:
        callback._shutdown_publisher()
    published = repository.latest()
    assert published == callback.last_receipt
    assert published["owner"] == repository.owner
    assert published["owner"][lineage] == options[lineage]
    assert published["checkpoint"]["step"] == 750
    assert client.objects[published["checkpoint"]["key"]] == path.read_bytes()
    assert json.loads(callback.receipt_path.read_text()) == published


@pytest.mark.integration
def test_training_advances_with_one_immutable_pending_file_and_drains_final(tmp_path, monkeypatch):
    release = threading.Event()
    started = threading.Event()
    callback = DurableStepCheckpoint(**callback_options(tmp_path))
    uploaded = []
    trained_while_pending = []

    def publish(path, step):
        original = path.read_bytes()
        state = torch.load(path, weights_only=True)
        assert state["global_step"] == step
        assert state["optimizer_states"][0]["state"]
        assert state["loops"]
        if step == 1:
            started.set()
            assert release.wait(10), "Training never advanced while upload was pending"
        assert path.read_bytes() == original
        uploaded.append(step)
        return receipt(step, path)

    monkeypatch.setattr(callback, "_publish", publish)

    class Model(TinyModel):
        def training_step(self, batch, batch_idx):
            if self.global_step in (1, 2, 3):
                assert started.wait(10)
                assert callback.last_receipt is None
                assert callback._pending_step == 1
                assert len(list((tmp_path / "recovery").glob("step_*.ckpt"))) == 1
                trained_while_pending.append(self.global_step)
            if self.global_step == 3:
                release.set()
            return super().training_step(batch, batch_idx)

    trainer = L.Trainer(
        max_steps=4, callbacks=[callback], limit_val_batches=0, **trainer_options(tmp_path)
    )
    try:
        trainer.fit(Model(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    finally:
        release.set()
    assert trained_while_pending == [1, 2, 3]
    assert uploaded == [1, 4]
    assert callback.last_receipt["checkpoint"]["step"] == 4
    assert callback._pending_step is None
    assert callback._executor is None
    files = list((tmp_path / "recovery").glob("step_*.ckpt"))
    assert len(files) == 1
    assert torch.load(files[0], weights_only=True)["global_step"] == 4
    assert torch.load(callback.last_model_path, weights_only=True)["global_step"] == 4


@pytest.mark.integration
def test_timer_stop_drains_final_before_validation(tmp_path, monkeypatch):
    callback = DurableStepCheckpoint(**callback_options(tmp_path))
    published = []
    seen_validation = []

    def publish(path, step):
        published.append(step)
        return receipt(step, path)

    monkeypatch.setattr(callback, "_publish", publish)

    class Stop(L.Callback):
        def on_train_batch_end(self, trainer, *args):
            if trainer.global_step == 3:
                trainer.should_stop = True

        def on_validation_start(self, trainer, pl_module):
            # Lightning orders checkpoint callbacks after ordinary callbacks,
            # proving this drain happened before any validation hook started.
            assert callback.last_receipt["checkpoint"]["step"] == trainer.global_step == 3
            seen_validation.append(trainer.global_step)

    options = trainer_options(tmp_path) | {"val_check_interval": 5000}
    trainer = L.Trainer(max_steps=10, callbacks=[Stop(), callback], **options)
    trainer.fit(
        TinyModel(),
        train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2),
        val_dataloaders=DataLoader([torch.ones(1)] * 2, batch_size=2),
    )
    assert seen_validation == [3]
    assert published[-1] == 3
    assert callback.last_receipt["checkpoint"]["step"] == 3


@pytest.mark.integration
def test_failed_background_upload_retains_full_local_recovery(tmp_path, monkeypatch, capsys):
    callback = DurableStepCheckpoint(**callback_options(tmp_path))

    def publish(path, step):
        assert torch.load(path, weights_only=True)["global_step"] == step
        raise TimeoutError("private SDK url with credentials")

    monkeypatch.setattr(callback, "_publish", publish)
    trainer = L.Trainer(
        max_steps=4, callbacks=[callback], limit_val_batches=0, **trainer_options(tmp_path)
    )
    with pytest.raises(RuntimeError, match=r"publication failed \(TimeoutError\)"):
        trainer.fit(TinyModel(), train_dataloaders=DataLoader(InfiniteOnes(), batch_size=2))
    assert callback.last_receipt is None
    assert not callback.receipt_path.exists()
    assert callback._executor is None
    files = list((tmp_path / "recovery").glob("step_*.ckpt"))
    assert len(files) == 1
    state = torch.load(files[0], weights_only=True)
    assert state["global_step"] == 1
    assert state["optimizer_states"][0]["state"]
    output = capsys.readouterr().out
    assert "checkpoint_upload_started" in output
    assert "checkpoint_published" not in output
    assert "credentials" not in output


class HeldPublication(DurableStepCheckpoint):
    def __init__(self, *, fail, **kwargs):
        super().__init__(**kwargs)
        self.release = threading.Event()
        self.started = threading.Event()
        self.fail = fail

    def _publish(self, path, step):
        self.started.set()
        assert self.release.wait(10)
        assert torch.load(path, weights_only=True) == {"global_step": step}
        if self.fail:
            raise TimeoutError("private SDK error")
        return receipt(step, path)


def _async_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    timeout = timedelta(seconds=30)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timeout
    )
    try:
        root = Path(directory)
        strategy = DDPStrategy(process_group_backend="gloo", timeout=timeout)

        def save_checkpoint(path, weights_only):
            assert weights_only is False
            if rank == 0:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"global_step": 1}, path)
            strategy.barrier()

        trainer = SimpleNamespace(
            is_global_zero=rank == 0,
            global_step=1,
            strategy=strategy,
            save_checkpoint=save_checkpoint,
            loggers=[],
        )
        reports = []
        for label, fail in (("success", False), ("failure", True)):
            callback = HeldPublication(fail=fail, **callback_options(root / label))
            callback._save_checkpoint(trainer, str(root / label / "recovery" / "step.ckpt"))
            callback._poll_publication(trainer)
            assert callback._pending_step == 1
            assert callback.last_receipt is None
            # Both ranks can perform a training-style collective while the
            # rank-zero uploader waits for the test to release it.
            value = torch.tensor(rank + 1)
            dist.all_reduce(value)
            assert value.item() == 3
            if rank == 0:
                assert callback.started.wait(10)
                callback.release.set()
            error = None
            try:
                callback._drain_publication(trainer)
            except RuntimeError as exception:
                error = str(exception)
            finally:
                callback.teardown(trainer, None, "fit")
            reports.append({"error": error, "receipt": callback.last_receipt})
        (root / f"rank-{rank}.json").write_text(json.dumps(reports))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires PyTorch Gloo backend")
def test_background_upload_allows_ddp_collectives_and_broadcasts_failures(tmp_path):
    context = mp.spawn(
        _async_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Background checkpoint test exceeded its bounded deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
    assert reports[0] == reports[1]
    success, failure = reports[0]
    assert success["error"] is None
    assert success["receipt"]["checkpoint"]["step"] == 1
    assert failure["receipt"] is None
    assert failure["error"] == "Durable checkpoint publication failed (TimeoutError); stopping"


@pytest.mark.parametrize("value", [1, None, "true"])
def test_async_setting_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="async_upload"):
        DurableStepCheckpoint(**(callback_options(tmp_path) | {"async_upload": value}))

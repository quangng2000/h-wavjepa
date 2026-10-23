"""External torchrun worker used by test_dynamics_ddp.py (not a pytest module)."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.train import train


def main():
    root, dataset, destination, accelerator, precision, variant = map(str, sys.argv[1:])
    rank = int(os.environ["RANK"])
    destination = Path(destination)
    for phase, max_steps in (("initial", 2), ("resumed", 4)):
        output = destination / phase
        with initialize_config_dir(version_base="1.3", config_dir=f"{root}/configs"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=upstream_smoke",
                    "data=wav",
                    f"paths.root_dir={root}",
                    f"paths.output_dir={output}",
                    f"paths.log_dir={destination}/logs",
                    f"paths.work_dir={root}",
                    f"data.train_manifest={dataset}/train.jsonl",
                    f"data.val_manifest={dataset}/val.jsonl",
                    f"data.test_manifest={dataset}/test.jsonl",
                ],
            )
        OmegaConf.set_struct(cfg, False)
        cfg.logger = None
        for key in ("train_size", "val_size", "test_size"):
            cfg.data.pop(key, None)
        cfg.extras = dict(ignore_warnings=False, enforce_tags=False, print_config=False)
        cfg.data.num_samples = 4608
        cfg.data.batch_size = 2
        cfg.data.minimum_batch_size = 2
        cfg.data.require_full_length = True
        cfg.data.train_drop_last = True
        cfg.data.train_random_crop = True
        cfg.data.num_workers = 0
        cfg.model.backbone.config.encoder_layers = 2
        cfg.model.backbone.config.average_top_k_layers = 2
        cfg.model.backbone.config.use_gradient_checkpointing = True
        cfg.model.readout_mode = "normalized_top_k"
        if variant == "temporal":
            cfg.model.aggregation_windows = [1, 2, 2]
            cfg.model.temporal_variance_weight = 0.1
            cfg.model.temporal_std_floor = 0.1
            cfg.model.temporal_variance_epsilon = 1e-4
        cfg.model.localereg_clusters = 2
        cfg.model.localereg_min_cluster_size = 2
        cfg.model.localereg_n_quad = 32
        cfg.model.lr_total_steps = 4
        cfg.model.ema_anneal_end_step = 4
        cfg.trainer.devices = 2
        cfg.trainer.accelerator = accelerator
        cfg.trainer.deterministic = accelerator == "cpu"
        cfg.trainer.strategy = "ddp_find_unused_parameters_false"
        cfg.trainer.use_distributed_sampler = False
        cfg.trainer.precision = precision
        cfg.trainer.accumulate_grad_batches = 2
        cfg.trainer.max_steps = max_steps
        # This is a step-budget experiment, as is the AudioSet preset. A max_epochs
        # cap can terminate a resumed Lightning loop before the desired step budget.
        cfg.trainer.max_epochs = -1
        cfg.trainer.limit_train_batches = 4
        cfg.trainer.limit_val_batches = 1
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.enable_model_summary = False
        cfg.callbacks.early_stopping = None
        if phase == "resumed":
            cfg.ckpt_path = str(destination / "initial" / "checkpoints" / "last.ckpt")
        metrics, objects = train(cfg)
        trainer, model, data = (objects[key] for key in ("trainer", "model", "datamodule"))
        assert trainer.global_step == max_steps
        assert model.ema_updates.item() == max_steps
        assert model._resolved_training_schedule["total_steps"] == 4
        assert model.readout_mode == "normalized_top_k"
        if variant == "temporal":
            assert model.config["aggregation_windows"] == [1, 2, 2]
            assert model.config["temporal_variance_weight"] == 0.1
            assert torch.isfinite(metrics["train/temporal_variance_loss"])
            assert torch.isfinite(metrics["val/temporal_variance_loss"])
        assert model.backbone.model.encoder is not model.backbone.model.teacher_encoder
        assert torch.isfinite(metrics["train/loss"])
        assert torch.isfinite(metrics["val/loss"])
        sampler = trainer.train_dataloader.batch_sampler.sampler
        assert sampler.num_replicas == 2 and sampler.rank == rank
        assert len(sampler) == 8
        assert data.data_train.random_crop
        state = model.state_dict()
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            value = tensor.detach().cpu().contiguous()
            assert torch.isfinite(value).all(), name
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
        report = {
            "rank": rank,
            "precision": str(trainer.precision),
            "global_step": trainer.global_step,
            "ema_updates": int(model.ema_updates),
            "state_sha256": digest.hexdigest(),
            "sample_indices": [list(index) for index in sampler],
            "sampler_epoch": sampler.epoch,
            "train_loss": float(metrics["train/loss"]),
            "val_loss": float(metrics["val/loss"]),
            "variant": variant,
        }
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

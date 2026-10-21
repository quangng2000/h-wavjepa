"""Hydra object construction and local, non-networked run provenance."""

import json
import platform
import subprocess
from pathlib import Path

import hydra
import lightning
import torch
from omegaconf import DictConfig, OmegaConf

# Sweep directories are resolved before HydraConfig exists. A source-root
# resolver is available at composition time and never depends on job state.
OmegaConf.register_new_resolver(
    "hwavjepa_root", lambda: str(Path(__file__).resolve().parents[2]), replace=True
)


def instantiate_collection(config: DictConfig | None) -> list:
    """Instantiate named callbacks/loggers, ignoring disabled entries."""
    if config is None:
        return []
    return [hydra.utils.instantiate(value) for value in config.values() if value is not None]


def write_provenance(cfg: DictConfig, model: lightning.LightningModule) -> None:
    """Persist the resolved configuration and versions; never upload them."""
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cfg.paths.root_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=cfg.paths.root_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = {
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "lightning": lightning.__version__,
        "hydra": hydra.__version__,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters_including_teacher": sum(p.numel() for p in model.parameters()),
        "seed": cfg.seed,
        "status": "research scaffold; metrics are not evidence of superiority",
    }
    (output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_run(cfg: DictConfig) -> tuple:
    lightning.seed_everything(cfg.seed, workers=True)
    if cfg.get("num_threads") is not None:
        torch.set_num_threads(cfg.num_threads)
    data = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = instantiate_collection(cfg.get("callbacks"))
    loggers = instantiate_collection(cfg.get("logger"))
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers or False,
        default_root_dir=cfg.paths.output_dir,
    )
    if trainer.is_global_zero:
        write_provenance(cfg, model)
    return data, model, trainer

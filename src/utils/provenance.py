"""H-WavJEPA extension: local, non-networked run provenance."""

import json
import platform
import subprocess
from pathlib import Path

import hydra
import lightning
import torch
from omegaconf import DictConfig, OmegaConf


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

"""Evaluate a trusted local checkpoint using matching Hydra configuration."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.utils.runtime import build_run


def evaluate(cfg: DictConfig) -> tuple:
    if not cfg.get("ckpt_path") or not Path(cfg.ckpt_path).is_file():
        raise ValueError("Evaluation requires ckpt_path=/absolute/path/to/trusted.ckpt")
    data, model, trainer = build_run(cfg)
    trainer.test(model, datamodule=data, ckpt_path=cfg.ckpt_path)
    return model, trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()

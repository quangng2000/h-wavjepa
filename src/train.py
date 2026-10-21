"""Run from the project root: uv run python -m src.train experiment=smoke."""

import hydra
from omegaconf import DictConfig

from src.utils.runtime import build_run


def train(cfg: DictConfig) -> tuple:
    data, model, trainer = build_run(cfg)
    trainer.fit(model, datamodule=data, ckpt_path=cfg.get("ckpt_path"))
    # Held-out test evaluation is opt-in; never select experiments on this split.
    if cfg.test:
        callback = trainer.checkpoint_callback
        best = callback.best_model_path if callback else None
        trainer.test(model, datamodule=data, ckpt_path=best or None)
    return model, trainer


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

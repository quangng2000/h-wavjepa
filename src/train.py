from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402 -- rootutils must establish the project import path first
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    write_provenance,
)
from src.utils.runtime import model_config_for_checkpoint, record_run, validate_config

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
@record_run
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    validate_config(cfg)
    extras(cfg)
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    if cfg.get("num_threads") is not None:
        torch.set_num_threads(cfg.num_threads)

    # Only load checkpoints the caller explicitly trusts, from local files.
    # Keep Lightning/PyTorch safe-loading defaults; do not disable weights_only.
    checkpoint = cfg.get("ckpt_path")
    if checkpoint and not Path(checkpoint).is_file():
        raise ValueError("ckpt_path must name an existing trusted local checkpoint")
    if cfg.get("test") and not cfg.get("train") and not checkpoint:
        raise ValueError("train=false,test=true requires an explicit trusted ckpt_path")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_kwargs = (
        {"input_sample_rate": cfg.data.sample_rate}
        if cfg.model._target_ == "src.models.hwavjepa_module.HWavJEPAModule"
        else {}
    )
    model: LightningModule = hydra.utils.instantiate(
        model_config_for_checkpoint(cfg), **model_kwargs
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger or False,
        default_root_dir=cfg.paths.output_dir,
    )
    if trainer.is_global_zero:
        write_provenance(cfg, model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=True
        )

    train_metrics = dict(trainer.callback_metrics)

    if cfg.get("test"):
        log.info("Starting testing!")
        # When training was skipped, evaluate the explicitly requested checkpoint.
        # Otherwise prefer the best checkpoint, or current weights when checkpointing
        # is disabled or no validation checkpoint was written.
        checkpoint_callback = trainer.checkpoint_callback
        ckpt_path = (
            checkpoint_callback.best_model_path or None
            if cfg.get("train") and checkpoint_callback is not None
            else None
        )
        if not cfg.get("train"):
            ckpt_path = checkpoint
        elif ckpt_path is None:
            log.warning("Best ckpt not found! Using current weights for testing...")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = dict(trainer.callback_metrics)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

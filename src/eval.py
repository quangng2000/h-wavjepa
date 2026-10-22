from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
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
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    write_provenance,
)
from src.utils.runtime import model_config_for_checkpoint, record_run, validate_config

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
@record_run
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    validate_config(cfg)
    extras(cfg)
    if not cfg.get("ckpt_path") or not Path(cfg.ckpt_path).is_file():
        raise ValueError("Evaluation requires ckpt_path=/absolute/path/to/trusted.ckpt")
    # Keep Lightning/PyTorch safe-loading defaults for the trusted local checkpoint.
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    if cfg.get("num_threads") is not None:
        torch.set_num_threads(cfg.num_threads)

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

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger or False, default_root_dir=cfg.paths.output_dir
    )
    if trainer.is_global_zero:
        write_provenance(cfg, model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=True)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = dict(trainer.callback_metrics)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    evaluate(cfg)


if __name__ == "__main__":
    main()

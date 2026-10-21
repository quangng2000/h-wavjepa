import hydra
import pytest
from omegaconf import OmegaConf

from src.utils import instantiate_callbacks, instantiate_loggers


@pytest.mark.parametrize(
    ("experiment", "levels", "protection"),
    [
        ("smoke", 3, 1.0),
        ("hwavjepa", 3, 1.0),
        ("hwavjepa_unprotected", 3, 0.0),
        ("two_level", 2, 1.0),
        ("flat", 1, 0.0),
        ("flat_protected", 1, 1.0),
    ],
)
def test_experiment_composes(make_config, experiment, levels, protection):
    cfg = make_config([f"experiment={experiment}"])
    assert len(cfg.model.encoder.widths) == levels
    assert len(cfg.model.prediction_weights) == levels
    assert cfg.model.protection_weight == protection
    assert cfg.model.mask_group_size == 4
    assert hydra.utils.instantiate(cfg.model) is not None


@pytest.mark.parametrize("trainer", ["cpu", "mps", "gpu"])
def test_trainer_config_composes(make_config, trainer):
    cfg = make_config([f"trainer={trainer}"])
    assert cfg.trainer.accelerator == trainer


def test_zero_seed_is_valid(make_config):
    cfg = make_config(["seed=0"])
    assert cfg.data.seed == 0


@pytest.mark.parametrize("name", ["train", "eval"])
def test_upstream_config_instantiation(make_config, name):
    cfg = make_config(name=name)
    assert hydra.utils.instantiate(cfg.data) is not None
    assert hydra.utils.instantiate(cfg.model) is not None
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
    assert trainer.default_root_dir == cfg.paths.output_dir
    assert cfg.paths.data_dir == f"{cfg.paths.root_dir}/data/"


def test_upstream_disabled_collections(make_config):
    cfg = make_config(["callbacks=none", "logger=[]"])
    assert instantiate_callbacks(cfg.callbacks) == []
    assert instantiate_loggers(cfg.get("logger")) == []


@pytest.mark.parametrize("factory", [instantiate_callbacks, instantiate_loggers])
def test_upstream_collection_validation(factory):
    assert factory(None) == []
    assert factory(OmegaConf.create({"disabled": None, "metadata": {"name": "no target"}})) == []
    with pytest.raises(TypeError, match="DictConfig"):
        factory([{"_target_": "must.not.be.imported"}])


def test_debug_fdr_composes_upstream_overrides(make_config):
    cfg = make_config(["debug=fdr"])
    assert cfg.task_name == "debug"
    assert cfg.trainer.fast_dev_run is True
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.callbacks is None
    assert cfg.logger is None

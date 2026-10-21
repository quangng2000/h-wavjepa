"""Do not publish inaccurate run records from parent-only multiprocessing state."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf, open_dict

from src.train import train
from src.utils.runtime import validate_config


@pytest.mark.parametrize(
    "strategy",
    [
        "ddp_spawn",
        "ddp_spawn_find_unused_parameters_true",
        "ddp_spawn_find_unused_parameters_false",
        "ddp_fork",
        "ddp_fork_find_unused_parameters_true",
        "ddp_fork_find_unused_parameters_false",
        "ddp_notebook",
        "ddp_notebook_find_unused_parameters_true",
        "ddp_notebook_find_unused_parameters_false",
    ],
)
def test_parent_only_distributed_aliases_are_rejected(make_config, strategy):
    cfg = make_config([f"+trainer.strategy={strategy}"])
    with pytest.raises(ValueError, match="externally launched DDP"):
        validate_config(cfg)


@pytest.mark.parametrize("start_method", ["spawn", "fork", "forkserver"])
@pytest.mark.parametrize(
    "target",
    [
        "lightning.pytorch.strategies.DDPStrategy",
        "lightning.pytorch.strategies.ddp.DDPStrategy",
    ],
)
def test_parent_only_distributed_strategy_objects_are_rejected(make_config, target, start_method):
    cfg = make_config()
    with open_dict(cfg.trainer):
        cfg.trainer.strategy = OmegaConf.create({"_target_": target, "start_method": start_method})
    with pytest.raises(ValueError, match="stale progress and checkpoint paths"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "strategy",
    ["auto", "ddp", "ddp_find_unused_parameters_true", "ddp_find_unused_parameters_false"],
)
def test_other_strategy_aliases_are_not_claimed_unsupported(make_config, strategy):
    # Passing preflight is not a distributed execution or hardware certification.
    validate_config(make_config([f"+trainer.strategy={strategy}"]))


@pytest.mark.parametrize("explicit_popen", [False, True])
def test_popen_ddp_strategy_configuration_remains_available(make_config, explicit_popen):
    cfg = make_config()
    specification = {"_target_": "lightning.pytorch.strategies.DDPStrategy"}
    if explicit_popen:
        specification["start_method"] = "popen"
    with open_dict(cfg.trainer):
        cfg.trainer.strategy = OmegaConf.create(specification)
    validate_config(cfg)


def test_inherited_ddp_sim_fails_before_training_and_records_failure(make_config, monkeypatch):
    cfg = make_config(["trainer=ddp_sim"])

    def unexpected_instantiate(*args, **kwargs):
        pytest.fail("Unsupported strategy must fail before model/trainer instantiation")

    monkeypatch.setattr("hydra.utils.instantiate", unexpected_instantiate)
    with pytest.raises(ValueError, match="not supported by run artifact tracking"):
        train(cfg)
    output = Path(cfg.paths.output_dir)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert not (output / "metrics.json").exists()
    assert not (output / "checkpoints").exists()


def test_strategy_safety_also_applies_to_other_hydra_models(make_config):
    cfg = make_config(["+trainer.strategy=ddp_spawn"])
    cfg.model._target_ = "example.CustomModel"
    with pytest.raises(ValueError, match="not supported by run artifact tracking"):
        validate_config(cfg)

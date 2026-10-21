import hydra
import pytest


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

"""Public utility contracts inherited from the actual Lightning-Hydra template."""

import pytest
import torch
from omegaconf import OmegaConf

from src.utils import get_metric_value, task_wrapper


def test_optimization_metric_is_returned_and_invalid_name_fails():
    metrics = {"val/loss": torch.tensor(0.25)}
    assert get_metric_value(metrics, "val/loss") == 0.25
    assert get_metric_value(metrics, None) is None
    with pytest.raises(Exception, match="Metric value not found"):
        get_metric_value(metrics, "val/not_logged")


def test_task_wrapper_does_not_hide_failure(tmp_path):
    @task_wrapper
    def failing_task(cfg):
        raise RuntimeError("deliberate task failure")

    with pytest.raises(RuntimeError, match="deliberate task failure"):
        failing_task(OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}))

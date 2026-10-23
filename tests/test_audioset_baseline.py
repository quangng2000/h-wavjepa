"""AudioSet controls isolate the added forecasting and regularization objectives."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.runtime import validate_config

PRESETS = [
    ("audioset_baseline", "audioset"),
    ("audioset_baseline_h100", "audioset_h100"),
    ("audioset_baseline_h100_fast", "audioset_h100_fast"),
    ("audioset_baseline_h100_four", "audioset_h100_four"),
]


@pytest.mark.parametrize(("baseline", "comparator"), PRESETS)
def test_audioset_masked_control_changes_only_the_added_objectives(
    make_config, baseline, comparator
):
    control = make_config([f"experiment={baseline}"])
    candidate = make_config([f"experiment={comparator}"])
    validate_config(control)
    validate_config(candidate)
    assert "masked-only-control" in control.tags
    assert "performance-measured" not in control.tags
    assert control.model.hierarchy is False
    assert list(control.model.prediction_weights) == [0.0, 0.0, 0.0]
    assert control.model.regularization_weight == 0.0
    assert control.model.masked_weight == 1.0
    assert control.data.num_samples == 9 * control.model.backbone.config.window_samples == 145440
    assert control.model.lr_total_steps == control.trainer.max_steps == 375000
    assert control.model.lr_warmup_steps == control.model.ema_anneal_end_step == 100000
    if control.trainer.accelerator == "gpu":
        assert control.data.batch_size * control.trainer.devices == 64
        assert control.trainer.get("accumulate_grad_batches", 1) == 1

    # Comparing the entire resolved configuration guards crop/sampler, masked
    # objective, initialization, precision, optimizer and validation parity.
    control_config = OmegaConf.to_container(control, resolve=True)
    candidate_config = OmegaConf.to_container(candidate, resolve=True)
    control_config.pop("tags")
    candidate_config.pop("tags")
    for field in ("hierarchy", "prediction_weights", "regularization_weight"):
        control_config["model"][field] = candidate_config["model"][field]
    assert control_config == candidate_config


@pytest.mark.parametrize(("baseline", "comparator"), PRESETS)
def test_audioset_masked_control_matches_upstream_loss_and_gradients(
    make_config, baseline, comparator
):
    cfg = make_config([f"experiment={baseline}"])
    tiny = make_config(["experiment=upstream_smoke"])
    # Shrink only model dimensions/windows so the actual preset objective and
    # its compact/checkpointed execution path can be checked without a GPU.
    model_config = deepcopy(cfg.model)
    tiny_backbone = deepcopy(tiny.model.backbone)
    for field in ("compact_masked_tokens", "use_gradient_checkpointing"):
        tiny_backbone.config[field] = cfg.model.backbone.config[field]
    model_config.backbone = tiny_backbone
    model_config.hidden_dim = tiny.model.hidden_dim
    model_config.predictor_layers = tiny.model.predictor_layers
    model_config.predictor_heads = tiny.model.predictor_heads
    model = instantiate(model_config).train()
    reference = deepcopy(model.backbone).train()
    audio = torch.randn(2, model.required_samples)

    # The baseline must avoid all extra temporal/regularizer computation rather
    # than merely multiplying unwanted terms by zero after computing them.
    with (
        patch.object(model, "encode", side_effect=AssertionError("forecast path ran")),
        patch.object(model.regularizer, "forward", side_effect=AssertionError("regularizer ran")),
    ):
        actual = model.compute_loss(audio, seed=23)
    expected = reference.masked_loss(audio.reshape(2, 9, model.window_samples), seed=23)
    torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["masked_loss"], expected, rtol=0, atol=0)
    assert actual["prediction_loss"].item() == actual["regularization_loss"].item() == 0
    assert actual["levels"] == actual["predictions"] == actual["targets"] == []
    assert list(model.abstractions.parameters()) == list(model.predictors.parameters()) == []

    actual["loss"].backward()
    expected.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.backbone.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, reference_parameters[name].grad, rtol=0, atol=0
            )
        else:
            assert parameter.grad is None, name

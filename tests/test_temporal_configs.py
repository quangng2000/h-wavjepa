"""Temporal experiments preserve data, optimization and execution controls."""

import pytest
from omegaconf import OmegaConf

from src.utils.runtime import validate_config


@pytest.mark.parametrize(
    ("temporal", "pointwise"),
    [
        ("audioset_temporal", "audioset"),
        ("audioset_temporal_h100", "audioset_h100"),
        ("audioset_temporal_h100_fast", "audioset_h100_fast"),
        ("audioset_temporal_h100_four", "audioset_h100_four"),
        ("upstream_temporal_smoke", "upstream_smoke"),
    ],
)
def test_temporal_presets_change_only_aggregation_and_temporal_pressure(
    make_config, temporal, pointwise
):
    candidate = make_config([f"experiment={temporal}"])
    baseline = make_config([f"experiment={pointwise}"])
    validate_config(candidate)
    validate_config(baseline)
    assert list(candidate.model.aggregation_windows) == [1, 2, 2]
    assert candidate.model.temporal_variance_weight == 0.1
    assert list(baseline.model.aggregation_windows) == [1, 1, 1]
    assert baseline.model.temporal_variance_weight == 0.0
    assert "performance-measured" not in candidate.tags
    left = OmegaConf.to_container(candidate, resolve=True)
    right = OmegaConf.to_container(baseline, resolve=True)
    for config in (left, right):
        config.pop("tags")
        config["model"].pop("aggregation_windows")
        config["model"].pop("temporal_variance_weight")
    assert left == right


@pytest.mark.parametrize("aggregation", ["[1,1,1]", "[1,2,2]"])
@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_factorial_four_h100_controls_keep_data_and_batch_budget(make_config, aggregation, weight):
    cfg = make_config(
        [
            "experiment=audioset_temporal_h100_four",
            f"model.aggregation_windows={aggregation}",
            f"model.temporal_variance_weight={weight}",
        ]
    )
    validate_config(cfg)
    assert cfg.data.num_samples == 145440
    assert cfg.data.batch_size * cfg.trainer.devices == 64
    assert cfg.model.lr_total_steps == cfg.trainer.max_steps == 375000
    assert cfg.model.masked_weight == 1.0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.aggregation_windows=[1,2]", "aggregation_windows"),
        ("model.aggregation_windows=[2,2,2]", "aggregation_windows"),
        ("model.aggregation_windows=[1,0,2]", "aggregation_windows"),
        ("model.aggregation_windows=[1,true,2]", "aggregation_windows"),
        ("model.aggregation_windows=[1,2,4]", "aggregation support"),
        ("model.aggregation_windows=[1,3,2]", "at least two valid"),
        ("model.temporal_variance_weight=-0.1", "nonnegative weight"),
        ("model.temporal_variance_weight=true", "finite number"),
        ("model.temporal_std_floor=0.01", "temporal_std_floor"),
        ("model.temporal_variance_epsilon=0", "positive epsilon"),
        ("model.hierarchy=false", "learned, active upper"),
        ("model.prediction_weights=[1,0,1]", "learned, active upper"),
    ],
)
def test_invalid_temporal_configs_fail_before_data_or_model_loading(make_config, override, message):
    cfg = make_config(["experiment=audioset_temporal", override])
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)

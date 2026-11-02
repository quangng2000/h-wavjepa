"""Conv presets retain their mean-pooling controls' training contracts."""

import pytest
from omegaconf import OmegaConf

from src.utils.runtime import validate_config


@pytest.mark.parametrize(
    ("candidate", "baseline"),
    [
        ("upstream_conv1d_smoke", "upstream_temporal_smoke"),
        ("audioset_conv1d", "audioset_temporal"),
        ("audioset_conv1d_h100_four", "audioset_temporal_h100_four"),
        ("audioset_nat_conv1d", "audioset_nat"),
        ("audioset_nat_conv1d_h100_four", "audioset_nat_h100_four"),
    ],
)
def test_conv_presets_change_only_aggregation_mode_and_tags(make_config, candidate, baseline):
    overrides = (
        ["data.index_path=/test/index.json", "data.naturalistic_manifest=/test/nat.json"]
        if "nat" in candidate
        else []
    )
    configs = [make_config([f"experiment={name}", *overrides]) for name in (candidate, baseline)]
    for config in configs:
        validate_config(config)
    assert configs[0].model.aggregation_mode == "conv1d"
    assert configs[1].model.aggregation_mode == "mean"
    assert "unbenchmarked" in configs[0].tags
    assert list(configs[0].model.aggregation_windows) == [1, 2, 2]
    resolved = [OmegaConf.to_container(config, resolve=True) for config in configs]
    for config in resolved:
        config.pop("tags")
        config["model"].pop("aggregation_mode")
    assert resolved[0] == resolved[1]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("model.aggregation_mode=unknown", "aggregation_mode"),
        ("model.aggregation_mode=true", "aggregation_mode"),
        ("model.hierarchy=false", "conv1d aggregation requires hierarchy=true"),
    ],
)
def test_invalid_conv_config_is_rejected_before_loading(make_config, override, message):
    config = make_config(["experiment=upstream_conv1d_smoke", override])
    with pytest.raises(ValueError, match=message):
        validate_config(config)

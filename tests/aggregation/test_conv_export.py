"""Learned kernels keep continuous causal context during batched extraction."""

import json

import pytest
import torch
from test_hear import audio, make_task
from test_hierarchical_dynamics import model_factory

from src.hear_api import HEARModel, get_scene_embeddings, get_timestamp_embeddings
from src.hear_eval import export_task


def learned_encoder():
    encoder = model_factory(aggregation_mode="conv1d", aggregation_windows=[1, 2, 2]).eval()
    with torch.no_grad():
        for abstraction in encoder.abstractions:
            abstraction[0].conv.weight.normal_(std=0.1)
    return encoder


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("representation,index", [("medium", 1), ("high", 2)])
def test_learned_convolutions_match_dense_output_across_hear_chunks(
    batch_size, representation, index
):
    encoder = learned_encoder()
    inputs = audio(512 * 9)
    expected = encoder.encode(inputs)[index].detach()
    model = HEARModel(encoder, representation, window_batch_size=batch_size)
    actual, _ = get_timestamp_embeddings(inputs, model)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(get_scene_embeddings(inputs, model), expected.mean(1))


def test_hear_metadata_identifies_learned_aggregation(tmp_path):
    model = HEARModel(learned_encoder(), "high", window_batch_size=2)
    output = export_task(model, make_task(tmp_path), tmp_path / "export")
    metadata = json.loads((output / "h-wavjepa-export.json").read_text())
    assert metadata["aggregation_mode"] == "conv1d"
    assert metadata["aggregation_windows"] == [1, 2, 2]
    assert metadata["level_support_windows"] == [1, 2, 4]

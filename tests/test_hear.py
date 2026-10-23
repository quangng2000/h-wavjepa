"""HEAR geometry, extraction, and export contracts (no benchmark claims)."""

import json
from copy import deepcopy

import hydra
import numpy as np
import pytest
import torch

from src.hear_api import HEARModel, get_scene_embeddings, get_timestamp_embeddings, load_model
from src.hear_eval import export_task


@pytest.fixture
def encoder(make_config):
    config = make_config(["experiment=upstream_smoke"])
    return hydra.utils.instantiate(config.model).eval()


@pytest.fixture
def multilayer_encoder(make_config):
    config = make_config(["experiment=upstream_smoke"])
    config.model.backbone.config.encoder_layers = 3
    config.model.backbone.config.average_top_k_layers = 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(103)
        encoder = hydra.utils.instantiate(config.model).eval()
    # Random post-norm top-k=1 fixtures can hide the wrong readout: the final
    # encoder norm is nearly an identity at initialization. Give it a visibly
    # different affine transform as a trained checkpoint can have.
    with torch.no_grad():
        norm = encoder.backbone.model.encoder.norm
        norm.weight.copy_(torch.linspace(0.4, 1.6, norm.weight.numel()))
        norm.bias.copy_(torch.linspace(-0.7, 0.3, norm.bias.numel()))
    return encoder


def audio(samples, batch=2):
    return torch.rand(batch, samples) * 2 - 1


@pytest.mark.parametrize("representation,index", [("low", 0), ("medium", 1), ("high", 2)])
@pytest.mark.parametrize("readout_mode", ["legacy", "normalized_top_k"])
def test_scene_and_timestamp_levels_match_dense_encoder(
    multilayer_encoder, representation, index, readout_mode
):
    encoder = multilayer_encoder
    encoder.readout_mode = readout_mode
    model = HEARModel(encoder, representation, window_batch_size=2)
    inputs = audio(512 * 5)
    original = inputs.clone()
    expected = encoder.encode(inputs)[index].detach()
    alternative = "legacy" if readout_mode == "normalized_top_k" else "normalized_top_k"
    wrong_low = encoder.backbone.encode_windows(encoder._windows(inputs), readout_mode=alternative)
    wrong_levels = encoder._levels(wrong_low, encoder.abstractions)
    assert (expected - wrong_levels[index]).abs().max() > 0.01
    timestamps, times = get_timestamp_embeddings(inputs, model)
    torch.testing.assert_close(timestamps, expected)
    torch.testing.assert_close(get_scene_embeddings(inputs, model), expected.mean(1))
    expected_times = (torch.arange(5) * 512 + 255.5) * 1000 / 16000
    torch.testing.assert_close(times, expected_times.expand(2, -1))
    assert timestamps.dtype == times.dtype == torch.float32
    assert not timestamps.requires_grad
    assert timestamps.device.type == "cpu"
    torch.testing.assert_close(inputs, original)


def test_partial_scene_weighting_and_coarse_timestamp_tail(encoder):
    model = HEARModel(encoder, "high")
    inputs = audio(512 + 128)
    padded = torch.nn.functional.pad(inputs, (0, 384))
    expected = encoder.encode(padded)[2].detach()
    weighted = (expected[:, 0] * 512 + expected[:, 1] * 128) / 640
    torch.testing.assert_close(get_scene_embeddings(inputs, model), weighted)
    values, times = get_timestamp_embeddings(inputs, model)
    assert values.shape == (2, 1, 16)
    torch.testing.assert_close(values[:, 0], expected[:, 0])
    assert (times < inputs.shape[1] * 1000 / model.sample_rate).all()


@pytest.mark.parametrize("representation", ["low", "medium", "high", "backbone"])
def test_short_clips_are_finite_and_timestamps_inside_real_audio(encoder, representation):
    model = HEARModel(encoder, representation)
    inputs = audio(3)
    values, times = get_timestamp_embeddings(inputs, model)
    assert values.shape == (2, 1, 16)
    assert torch.isfinite(values).all()
    assert torch.isfinite(get_scene_embeddings(inputs, model)).all()
    torch.testing.assert_close(times, torch.full((2, 1), 1.0 / 16))


def test_backbone_grid_is_native_and_continuous_across_context_windows(encoder):
    model = HEARModel(encoder, "backbone", window_batch_size=1)
    inputs = audio(512 * 3 + 13)
    values, times = get_timestamp_embeddings(inputs, model)
    backbone = encoder.backbone
    count = (inputs.shape[1] - backbone.receptive_field_samples) // backbone.stride_samples + 1
    assert values.shape == (2, count, 16)
    assert times.dtype == torch.float32
    expected = (
        (torch.arange(count) * backbone.stride_samples + (backbone.receptive_field_samples - 1) / 2)
        * 1000
        / model.sample_rate
    )
    torch.testing.assert_close(times, expected.expand(2, -1))
    model.window_batch_size = 3
    other_values, other_times = get_timestamp_embeddings(inputs, model)
    torch.testing.assert_close(values, other_values)
    torch.testing.assert_close(times, other_times)
    # Fine tokens are real backbone outputs, not repeated window vectors.
    assert not torch.equal(values[:, 0], values[:, 1])


def test_window_batch_limit_applies_across_recordings(encoder, monkeypatch):
    model = HEARModel(encoder, "backbone", window_batch_size=2)
    calls = []
    original = encoder.backbone.encode_tokens

    def record(windows, **kwargs):
        calls.append(windows.shape[0])
        return original(windows, **kwargs)

    monkeypatch.setattr(encoder.backbone, "encode_tokens", record)
    get_scene_embeddings(audio(512 * 7, batch=3), model)
    assert calls and max(calls) <= 2


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros(0, 512),
        torch.zeros(2, 0),
        torch.zeros(2, 512).double(),
        torch.full((2, 512), float("nan")),
        torch.full((2, 512), 1.1),
    ],
)
def test_invalid_hear_audio_rejected(encoder, invalid):
    with pytest.raises(ValueError):
        get_scene_embeddings(invalid, HEARModel(encoder))


def test_loader_round_trip_and_option_validation(encoder, tmp_path):
    checkpoint = {"state_dict": deepcopy(encoder.state_dict())}
    encoder.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    loaded = load_model(path, representation="medium", window_batch_size=1)
    inputs = audio(1024)
    torch.testing.assert_close(
        get_scene_embeddings(inputs, loaded),
        get_scene_embeddings(inputs, HEARModel(encoder, "medium")),
    )
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="representation"):
        HEARModel(encoder, "all")
    with pytest.raises(ValueError, match="window_batch_size"):
        HEARModel(encoder, window_batch_size=0)
    loaded.train()
    with pytest.raises(ValueError, match="eval"):
        get_scene_embeddings(inputs, loaded)


@pytest.mark.parametrize("saved_mode", ["legacy", "normalized_top_k", None])
def test_hear_checkpoint_loading_preserves_readout_semantics(
    multilayer_encoder, saved_mode, tmp_path
):
    encoder = multilayer_encoder
    expected_mode = saved_mode or "legacy"
    encoder.readout_mode = expected_mode
    encoder.config["readout_mode"] = expected_mode
    checkpoint = {"state_dict": deepcopy(encoder.state_dict())}
    encoder.on_save_checkpoint(checkpoint)
    if saved_mode is None:
        checkpoint[encoder.CHECKPOINT_KEY]["dynamics"].pop("readout_mode")
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    inputs = audio(1024)
    expected = encoder.encode(inputs)
    for index, representation in enumerate(("low", "medium", "high")):
        loaded = load_model(path, representation=representation)
        assert loaded.encoder.readout_mode == expected_mode
        values, _ = get_timestamp_embeddings(inputs, loaded)
        torch.testing.assert_close(values, expected[index])


def make_task(tmp_path, embedding_type="scene"):
    sf = pytest.importorskip("soundfile")
    task = tmp_path / "task"
    task.mkdir()
    metadata = {"task_name": "test", "embedding_type": embedding_type, "splits": ["train"]}
    (task / "task_metadata.json").write_text(json.dumps(metadata))
    (task / "labelvocabulary.csv").write_text("idx,label\n0,tone\n")
    root = task / "16000" / "train"
    root.mkdir(parents=True)
    sf.write(root / "a.wav", np.linspace(-0.5, 0.5, 1024, dtype=np.float32), 16000)
    sf.write(root / "b.wav", np.linspace(-0.5, 0.5, 1300, dtype=np.float32), 16000)
    label = ["tone"] if embedding_type == "scene" else [{"start": 0, "end": 100, "label": "tone"}]
    (task / "train.json").write_text(json.dumps({"a.wav": label, "b.wav": label}))
    return task


@pytest.mark.parametrize("embedding_type", ["scene", "event"])
def test_export_preserves_variable_lengths_and_task_splits(encoder, tmp_path, embedding_type):
    task = make_task(tmp_path, embedding_type)
    output = export_task(HEARModel(encoder, "backbone"), task, tmp_path / "export")
    assert (output / "train.json").read_bytes() == (task / "train.json").read_bytes()
    manifest = json.loads((output / "h-wavjepa-export.json").read_text())
    assert manifest["protocol_version"] == 1
    assert manifest["temporal_grid"] == "native_tokens"
    assert manifest["timestamp_hop_samples"] == encoder.backbone.stride_samples
    assert manifest["training_state_strides"] == [1, 2, 4]
    assert manifest["readout_mode"] == encoder.readout_mode
    assert manifest["representation_readout"] == "final_layer_tokens"
    assert manifest["average_top_k_layers"] == encoder.backbone.config["average_top_k_layers"]
    assert manifest["hierarchy"] is True
    assert manifest["duplicate_levels"] == []
    assert "frontend support" in manifest["padding"]
    a = np.load(output / "train/a.wav.embedding.npy", allow_pickle=False)
    b = np.load(output / "train/b.wav.embedding.npy", allow_pickle=False)
    assert a.dtype == b.dtype == np.float32
    if embedding_type == "event":
        assert len(a) < len(b)
        times = json.loads((output / "train/b.wav.timestamps.json").read_text())
        assert len(times) == len(b)
        assert max(times) < 1300 * 1000 / 16000
    else:
        assert a.shape == b.shape == (16,)
    with pytest.raises(FileExistsError):
        export_task(HEARModel(encoder), task, output)


@pytest.mark.parametrize("hierarchy", [False, True])
@pytest.mark.parametrize("readout_mode", ["legacy", "normalized_top_k"])
def test_export_records_window_readout_and_flat_duplicates(
    make_config, tmp_path, hierarchy, readout_mode
):
    config = make_config(["experiment=upstream_smoke"])
    config.model.hierarchy = hierarchy
    config.model.readout_mode = readout_mode
    encoder = hydra.utils.instantiate(config.model).eval()
    output = export_task(HEARModel(encoder, "medium"), make_task(tmp_path), tmp_path / "export")
    manifest = json.loads((output / "h-wavjepa-export.json").read_text())
    assert manifest["readout_mode"] == manifest["representation_readout"] == readout_mode
    assert manifest["hierarchy"] is hierarchy
    assert manifest["duplicate_levels"] == ([] if hierarchy else ["low", "medium", "high"])
    if not hierarchy:
        inputs = audio(1024)
        low = get_scene_embeddings(inputs, HEARModel(encoder, "low"))
        for representation in ("medium", "high"):
            torch.testing.assert_close(
                get_scene_embeddings(inputs, HEARModel(encoder, representation)),
                low,
                rtol=0,
                atol=0,
            )


def test_export_rejects_path_escape(encoder, tmp_path):
    task = make_task(tmp_path)
    (task / "train.json").write_text(json.dumps({"../outside.wav": ["tone"]}))
    with pytest.raises(ValueError, match="escapes"):
        export_task(HEARModel(encoder), task, tmp_path / "export")

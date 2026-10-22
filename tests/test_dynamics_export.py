"""Exercise the public extraction path against the dynamics checkpoint contract."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.dynamics_inference import load_model
from src.embed import extract_embeddings
from src.utils.runtime import model_config_for_checkpoint


@pytest.mark.parametrize("name", ["train", "eval"])
def test_default_command_selects_official_smoke(name):
    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
        cfg = compose(config_name=name)
    assert cfg.model._target_.endswith("HierarchicalWavJEPAModule")
    assert cfg.model.backbone.config.encoder_dim == 16
    assert cfg.data.num_samples == 4608


def test_pretrained_experiment_composes_without_downloading(make_config):
    cfg = make_config(["experiment=upstream_pretrained"])
    assert cfg.model.backbone.pretrained is True
    assert cfg.model.backbone.config.encoder_dim == 768


def test_resume_skips_initialization_download_and_file(make_config, monkeypatch):
    cfg = make_config(
        [
            "experiment=upstream_smoke",
            "ckpt_path=/checkpoint.ckpt",
            "model.backbone.pretrained=true",
        ]
    )
    cfg.model.backbone.checkpoint_path = "/missing-original-state.pt"
    cfg.model.backbone.pretrained_path = "/missing-original.safetensors"
    construction = model_config_for_checkpoint(cfg)
    model = instantiate(construction)
    assert model.backbone.checkpoint_sha256 is None
    assert cfg.model.backbone.pretrained is True
    assert cfg.model.backbone.checkpoint_path == "/missing-original-state.pt"


@pytest.fixture
def dynamics_export_fixture(make_config, tmp_path):
    cfg = make_config(["experiment=upstream_smoke"])
    model = instantiate(cfg.model).eval()
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    waveform = ((np.arange(4608) % 257) * 30 - 3000).astype("<i2")
    audio = tmp_path / "recording.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(waveform.tobytes())
    manifest = tmp_path / "recordings.jsonl"
    manifest.write_text(json.dumps({"path": audio.name}) + "\n")
    return path, manifest, model, torch.from_numpy(waveform.astype(np.float32) / 32768)[None]


def test_dynamics_loader_preserves_rng_and_disables_gradients(dynamics_export_fixture):
    path, _, original, waveform = dynamics_export_fixture
    before = torch.random.get_rng_state().clone()
    loaded = load_model(path)
    assert torch.equal(before, torch.random.get_rng_state())
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    with torch.no_grad():
        for expected, actual in zip(
            original.encode(waveform), loaded.encode(waveform), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_export_uses_dynamics_training_grid(dynamics_export_fixture, tmp_path):
    path, manifest, model, waveform = dynamics_export_fixture
    output = tmp_path / "embeddings"
    result = extract_embeddings(path, manifest, output, num_samples=4608, model_family="dynamics")
    assert result["status"] == "completed"
    assert result["model_family"] == "dynamics"
    assert result["temporal_grid"] == "training_strides"
    assert result["level_widths"] == [16, 16, 16]
    shard = torch.load(output / "batch-000000.pt", weights_only=True)
    assert shard["protected"] is None
    with torch.no_grad():
        for expected, actual in zip(
            model.encode(waveform, subsample=True), shard["levels"], strict=True
        ):
            torch.testing.assert_close(actual, expected.mean(dim=1), rtol=0, atol=0)
    assert json.loads((output / "index.jsonl").read_text())["row"] == 0


def test_dynamics_export_rejects_padded_audio(dynamics_export_fixture, tmp_path):
    path, manifest, _, _ = dynamics_export_fixture
    with pytest.raises(ValueError, match="padding cannot be used"):
        extract_embeddings(
            path, manifest, tmp_path / "embeddings", num_samples=5120, model_family="dynamics"
        )


def test_checkpoint_family_is_explicit(dynamics_export_fixture, tmp_path):
    path, manifest, _, _ = dynamics_export_fixture
    with pytest.raises(ValueError, match="missing H-WavJEPA"):
        extract_embeddings(path, manifest, tmp_path / "embeddings", num_samples=4608)

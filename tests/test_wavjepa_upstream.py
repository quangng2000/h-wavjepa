"""Integration checks use actual pinned upstream modules, never fake backbones."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.components.wavjepa_backbone import UPSTREAM_COMMIT, WavJEPABackbone
from src.vendor.wavjepa.jepa import JEPA
from src.vendor.wavjepa.masking import TimeInverseBlockMasker


@pytest.fixture
def tiny_config():
    return {
        "sample_rate": 16000,
        "window_samples": 512,
        "conv_layers": [[8, 4, 2], [8, 3, 2]],
        "encoder_dim": 16,
        "encoder_heads": 2,
        "encoder_layers": 1,
        "decoder_dim": 8,
        "decoder_heads": 2,
        "decoder_layers": 1,
        "average_top_k_layers": 1,
        "masker": {
            "target_masks_per_context": 1,
            "context_mask_prob": 0.25,
            "context_mask_length": 2,
            "target_prob": 0.25,
            "target_length": 2,
        },
    }


def test_encoder_is_official_and_differentiable(tiny_config):
    backbone = WavJEPABackbone(tiny_config)
    assert isinstance(backbone.model, JEPA)
    wave = torch.randn(2, 3, 512)
    embeddings = backbone.encode_windows(wave)
    assert embeddings.shape == (2, 3, 16)
    embeddings[..., 0].sum().backward()
    conv = next(backbone.model.extract_audio.parameters())
    encoder = next(backbone.model.encoder.parameters())
    assert conv.grad is not None and conv.grad.abs().sum() > 0
    assert encoder.grad is not None and encoder.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.model.teacher_encoder.parameters())


def test_complete_windows_have_no_future_dependency(tiny_config):
    backbone = WavJEPABackbone(tiny_config).eval()
    original = torch.randn(1, 4, 512)
    perturbed = original.clone()
    perturbed[:, 2:] = 100 * torch.randn_like(perturbed[:, 2:])
    torch.testing.assert_close(
        backbone.encode_windows(original)[:, :2],
        backbone.encode_windows(perturbed)[:, :2],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        backbone.encode_teacher_windows(original)[:, :2],
        backbone.encode_teacher_windows(perturbed)[:, :2],
        atol=0,
        rtol=0,
    )


def test_original_masked_objective_is_reused_and_seed_local(tiny_config):
    backbone = WavJEPABackbone(tiny_config).train()
    wave = torch.randn(2, 2, 512)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    loss = backbone.masked_loss(wave, seed=123)
    again = backbone.masked_loss(wave, seed=123)
    torch.testing.assert_close(loss, again, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), torch_before)
    np.testing.assert_equal(np.random.get_state(), numpy_before)
    masks = backbone.masker(
        4, backbone.model.total_patches, 1, generator=np.random.default_rng(123)
    )
    expected = backbone.model(backbone._normalized_windows(wave), *masks)["loss"]
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert next(backbone.model.encoder.parameters()).grad is not None


def test_teacher_readout_and_update_use_official_teacher(tiny_config):
    backbone = WavJEPABackbone(tiny_config).train()
    assert not backbone.model.teacher_encoder.training
    backbone.model.train()
    assert not backbone.model.teacher_encoder.training
    target = backbone.encode_teacher_windows(torch.randn(2, 3, 512))
    assert target.shape == (2, 3, 16) and not target.requires_grad
    teacher = next(backbone.model.teacher_encoder.parameters())
    student = next(backbone.model.encoder.parameters())
    before = teacher.clone()
    with torch.no_grad():
        student.add_(1)
    backbone.update_masked_teacher(0.75)
    torch.testing.assert_close(teacher, before * 0.75 + student * 0.25)


def test_training_masks_follow_torch_seed_instead_of_os_entropy(tiny_config):
    backbone = WavJEPABackbone(tiny_config).train()
    wave = torch.randn(2, 2, 512)
    torch.manual_seed(72)
    first = [backbone.masked_loss(wave) for _ in range(2)]
    torch.manual_seed(72)
    repeated = [backbone.masked_loss(wave) for _ in range(2)]
    for left, right in zip(first, repeated, strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert not torch.equal(first[0], first[1])


def test_checkpoint_strict_roundtrip_and_hash(tiny_config, tmp_path):
    original = WavJEPABackbone(tiny_config)
    path = tmp_path / "official.ckpt"
    torch.save({"state_dict": original.model.state_dict()}, path)
    restored = WavJEPABackbone(tiny_config, checkpoint_path=str(path))
    assert restored.checkpoint_sha256 is not None and len(restored.checkpoint_sha256) == 64
    for key, value in original.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0, rtol=0)
    config = restored.construction_config()
    config["masker"]["target_prob"] = 0.5
    assert restored.config["masker"]["target_prob"] == 0.25


def test_checkpoint_missing_keys_does_not_partially_mutate(tiny_config, tmp_path):
    backbone = WavJEPABackbone(tiny_config)
    before = {k: v.clone() for k, v in backbone.model.state_dict().items()}
    path = tmp_path / "bad.pt"
    torch.save({next(iter(before)): torch.zeros_like(next(iter(before.values())))}, path)
    with pytest.raises(ValueError, match="incompatible"):
        backbone.load_official_checkpoint(path)
    for key, value in before.items():
        torch.testing.assert_close(value, backbone.model.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize("field", ["compile_modules", "use_gradient_checkpointing"])
def test_unsupported_execution_modes_fail_closed(tiny_config, field):
    tiny_config[field] = True
    with pytest.raises(ValueError, match="unsupported"):
        WavJEPABackbone(tiny_config)


def test_masker_retry_is_bounded():
    masker = TimeInverseBlockMasker()
    with pytest.raises(RuntimeError, match="max_attempts"):
        masker(1, 100, 1, generator=np.random.default_rng(4), max_attempts=0)


def test_source_provenance_and_licenses_present():
    vendor = Path(__file__).resolve().parents[1] / "src" / "vendor" / "wavjepa"
    manifest = json.loads((vendor / "UPSTREAM.json").read_text())
    assert manifest["commit"] == UPSTREAM_COMMIT
    assert manifest["source_sha256"]["wavjepa/jepa.py"] == (
        "e89e13c56262a94bfa2233094880b33db5f31d1de83c927f51178909b9707e69"
    )
    assert "BSD 3-Clause" in (vendor / "LICENSE").read_text()
    assert "Attribution-NonCommercial" in (vendor / "LICENSE-AudioMAE").read_text()


@pytest.mark.parametrize("shape", [(2, 512), (2, 3, 511), (0, 3, 512)])
def test_invalid_window_shape_rejected(tiny_config, shape):
    backbone = WavJEPABackbone(tiny_config)
    with pytest.raises(ValueError):
        backbone.encode_windows(torch.randn(shape))


def test_real_default_preset_preserves_integer_window_size():
    # Exercise the full production configuration without allocating ~785 MB of
    # random weights; all convolutional output shapes are computed on meta.
    with torch.device("meta"):
        backbone = WavJEPABackbone()
    assert backbone.model.target_length == backbone.window_samples == 32160
    assert backbone.model.total_patches == 200
    assert backbone.stride_samples == 160
    assert backbone.receptive_field_samples == 240
    torch.testing.assert_close(
        backbone.token_timestamps_ms()[:2], torch.tensor([7.46875, 17.46875], dtype=torch.float64)
    )


def test_token_readout_and_teacher_grad_contract(tiny_config):
    backbone = WavJEPABackbone(tiny_config).eval()
    wave = torch.randn(2, 3, 512, requires_grad=True)
    tokens = backbone.encode_tokens(wave)
    assert tokens.shape == (2, 3, backbone.model.total_patches, 16)
    torch.testing.assert_close(tokens.mean(dim=2), backbone.encode_windows(wave))
    teacher = backbone.encode_tokens(wave, teacher=True)
    assert not teacher.requires_grad
    torch.testing.assert_close(teacher.mean(dim=2), backbone.encode_teacher_windows(wave))
    tokens.square().sum().backward()
    assert wave.grad is not None
    timestamps = backbone.token_timestamps_ms()
    assert timestamps[0] == (backbone.receptive_field_samples - 1) / 2 / 16000 * 1000
    assert torch.all(timestamps[1:] > timestamps[:-1])


def test_huggingface_safetensors_roundtrip(tiny_config, tmp_path):
    from safetensors.torch import save_file

    original = WavJEPABackbone(tiny_config).eval()
    path = tmp_path / "model.safetensors"
    save_file({f"model.{k}": v for k, v in original.model.state_dict().items()}, path)
    restored = WavJEPABackbone(tiny_config, pretrained_path=str(path)).eval()
    assert len(restored.checkpoint_sha256) == 64
    wave = torch.randn(2, 2, 512)
    torch.testing.assert_close(
        original.encode_tokens(wave), restored.encode_tokens(wave), atol=0, rtol=0
    )
    torch.testing.assert_close(
        original.masked_loss(wave, seed=42), restored.masked_loss(wave, seed=42), atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "corruption", ["missing", "extra", "shape", "nonfinite", "prefix", "dtype"]
)
def test_huggingface_invalid_state_never_partially_mutates(tiny_config, tmp_path, corruption):
    from safetensors.torch import save_file

    backbone = WavJEPABackbone(tiny_config)
    before = {k: v.clone() for k, v in backbone.model.state_dict().items()}
    state = {f"model.{k}": v.clone() for k, v in before.items()}
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["model.unexpected"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = torch.ones(1)
    elif corruption == "nonfinite":
        state[key].view(-1)[0] = float("nan")
    elif corruption == "prefix":
        state[key.removeprefix("model.")] = state.pop(key)
    else:
        state[key] = state[key].half()
    path = tmp_path / "invalid.safetensors"
    save_file(state, path)
    with pytest.raises(ValueError):
        backbone.load_huggingface_checkpoint(path)
    assert backbone.checkpoint_sha256 is None
    for key, value in before.items():
        torch.testing.assert_close(value, backbone.model.state_dict()[key], atol=0, rtol=0)


def test_compiled_checkpoint_collision_rejected(tiny_config, tmp_path):
    backbone = WavJEPABackbone(tiny_config)
    state = dict(backbone.model.state_dict())
    key = next(k for k in state if "." in k)
    state[key.replace(".", "._orig_mod.", 1)] = state[key].clone()
    path = tmp_path / "collision.ckpt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="collide"):
        backbone.load_official_checkpoint(path)


def test_pretrained_configuration_rejected_before_network(tiny_config, monkeypatch):
    def unexpected_download(**kwargs):
        pytest.fail("incompatible configuration must not download")

    monkeypatch.setattr(
        "src.models.components.wavjepa_backbone.download_official_weights", unexpected_download
    )
    with pytest.raises(ValueError, match="default backbone"):
        WavJEPABackbone(tiny_config, pretrained=True)
    with pytest.raises(ValueError, match="choose only one"):
        WavJEPABackbone(tiny_config, pretrained=True, checkpoint_path="anything.ckpt")

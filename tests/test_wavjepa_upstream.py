"""Integration checks use actual pinned upstream modules, never fake backbones."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.components.wavjepa_backbone import UPSTREAM_COMMIT, WavJEPABackbone
from src.vendor.wavjepa.jepa import JEPA
from src.vendor.wavjepa.masking import TimeInverseBlockMasker
from src.vendor.wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid


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


@pytest.mark.parametrize("field", ["compile_modules"])
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


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_matching_student_teacher_multilayer_readouts(tiny_config, top_k):
    # More than one layer is essential: one-layer fixtures miss this mismatch.
    tiny_config.update(encoder_layers=3, average_top_k_layers=top_k)
    backbone = WavJEPABackbone(tiny_config).train()
    wave = torch.randn(2, 3, 512, requires_grad=True)
    student = backbone.encode_windows(wave, readout_mode="normalized_top_k")
    teacher = backbone.encode_teacher_windows(wave)
    # Train/eval Transformer kernels can differ at float32 rounding scale.
    torch.testing.assert_close(student, teacher, atol=2e-7, rtol=1e-6)
    assert student.requires_grad and not teacher.requires_grad
    student[..., 0].square().sum().backward()
    assert wave.grad is not None and wave.grad.abs().sum() > 0
    for layer in backbone.model.encoder.layers:
        assert layer.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in backbone.model.teacher_encoder.parameters())
    if top_k > 1:
        assert not torch.allclose(student, backbone.encode_windows(wave, readout_mode="legacy"))


def test_reused_teacher_targets_are_exact_and_computed_once(tiny_config, monkeypatch):
    tiny_config.update(encoder_layers=3, average_top_k_layers=2)
    backbone = WavJEPABackbone(tiny_config).train()
    wave = torch.randn(2, 3, 512)
    expected = backbone.encode_teacher_windows(wave)
    calls = []
    original = backbone.model._forward_teacher

    def track_teacher(x):
        calls.append(x.shape[0])
        return original(x)

    monkeypatch.setattr(backbone.model, "_forward_teacher", track_teacher)
    loss, reused = backbone.masked_loss_with_targets(wave, seed=19)
    assert calls == [6]
    assert not reused.requires_grad
    torch.testing.assert_close(reused, expected, atol=0, rtol=0)
    loss.backward()
    assert backbone.model.encoder.layers[0].self_attn.in_proj_weight.grad is not None


def test_checkpointing_preserves_masks_outputs_and_parameter_gradients(tiny_config):
    tiny_config.update(encoder_layers=3, average_top_k_layers=2)
    ordinary = WavJEPABackbone(tiny_config).train()
    checkpointed = WavJEPABackbone({**tiny_config, "use_gradient_checkpointing": True}).train()
    checkpointed.load_state_dict(ordinary.state_dict())
    wave = torch.randn(2, 3, 512)
    losses = []
    for backbone in (ordinary, checkpointed):
        masked, targets = backbone.masked_loss_with_targets(wave, seed=31)
        online = backbone.encode_windows(wave, readout_mode="normalized_top_k")
        loss = masked + (online - targets).square().mean() + online[..., 0].square().mean()
        losses.append(loss)
        loss.backward()
    torch.testing.assert_close(losses[0], losses[1], atol=0, rtol=0)
    for (name, left), (other, right) in zip(
        ordinary.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert name == other
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_bfloat16_autocast_keeps_normalization_and_loss_float32(tiny_config, checkpointing):
    tiny_config.update(
        encoder_layers=3, average_top_k_layers=2, use_gradient_checkpointing=checkpointing
    )
    backbone = WavJEPABackbone(tiny_config).train()
    wave = torch.randn(2, 3, 512)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, targets = backbone.masked_loss_with_targets(wave, seed=31)
        online = backbone.encode_windows(wave, readout_mode="normalized_top_k")
        loss = loss + online[..., 0].square().mean()
    assert loss.dtype == targets.dtype == online.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize(
    ("width", "length", "digest"),
    [
        (768, 200, "cc3f12feab365f34202328040784b4a2db7fc9764b5b082cab2ac513885cedaa"),
        (384, 200, "89f483ecdfec2cba420c4bfa1baa35b857e1f3d27fa060235cc8dab2e9bd0410"),
        (16, 127, "320a5b0d65c9855333695fe78270c3c9e78d30fb08a3e4b6b1fd59472e827cbd"),
    ],
)
def test_independent_sinusoid_matches_original_source_float32_golden(width, length, digest):
    # Generated by unmodified wavjepa/pos_embed.py at UPSTREAM_COMMIT, whose
    # source SHA-256 is pinned in UPSTREAM.json; no integration helper was used.
    positions = get_1d_sincos_pos_embed_from_grid(width, np.arange(length, dtype=np.float64))
    assert hashlib.sha256(positions.astype("<f4").tobytes()).hexdigest() == digest


def test_official_position_adaptation_changes_no_learned_tensor():
    # A meta model checks the production layout without allocating a full model.
    with torch.device("meta"):
        backbone = WavJEPABackbone({"window_samples": 16160, "use_gradient_checkpointing": True})
    state = dict(backbone.model.state_dict())
    for key, width in (("pos_encoding_encoder", 768), ("pos_encoding_decoder", 384)):
        state[key] = (
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(width, np.arange(200)))
            .float()
            .unsqueeze(0)
        )
    adapted = backbone._adapt_fixed_positions(state)
    assert backbone.model.total_patches == 100
    for key in state:
        if key.startswith("pos_encoding_"):
            assert adapted[key].shape[1] == 100
        else:
            assert adapted[key] is state[key]
    corrupted = dict(state)
    corrupted["pos_encoding_encoder"] = state["pos_encoding_encoder"].clone()
    corrupted["pos_encoding_encoder"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="exact official"):
        backbone._adapt_fixed_positions(corrupted)


def test_official_pretrained_accepts_window_and_checkpointing_flags_before_download(monkeypatch):
    # Stop at download after checking the real base architecture on meta.
    def downloaded(**kwargs):
        raise RuntimeError("download reached")

    monkeypatch.setattr(
        "src.models.components.wavjepa_backbone.download_official_weights", downloaded
    )
    with torch.device("meta"), pytest.raises(RuntimeError, match="download reached"):
        WavJEPABackbone(
            {"window_samples": 16160, "use_gradient_checkpointing": True}, pretrained=True
        )

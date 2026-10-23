"""Binaural architecture, channel alignment and independent-crop masked loss."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.vendor.wavjepa.extractors import ConvChannelFeatureExtractor
from src.vendor.wavjepa.pos_embed import get_binaural_sincos_pos_embed


def nat_config(**overrides):
    return {
        "input_channels": 2,
        "window_samples": 128,
        "conv_layers": [[8, 4, 2], [8, 3, 2]],
        "encoder_dim": 16,
        "encoder_heads": 2,
        "encoder_layers": 2,
        "decoder_dim": 8,
        "decoder_heads": 2,
        "decoder_layers": 2,
        "average_top_k_layers": 2,
        "masker": {
            "target_masks_per_context": 2,
            "context_mask_prob": 0.5,
            "context_mask_length": 2,
            "target_prob": 0.25,
            "target_length": 2,
        },
        **overrides,
    }


def test_official_channel_extractor_has_independent_ears_in_channel_major_order():
    model = WavJEPABackbone(nat_config()).eval()
    extractor = model.model.extract_audio
    assert isinstance(extractor, ConvChannelFeatureExtractor)
    assert extractor.weight_sharing is False
    assert len(extractor.cnns) == 2
    left = next(extractor.cnns[0].parameters())
    right = next(extractor.cnns[1].parameters())
    assert left.data_ptr() != right.data_ptr()
    assert not torch.equal(left, right)
    wave = torch.randn(2, 2, 128)
    actual = extractor(wave)
    expected = torch.cat(
        [extractor.cnns[i](wave[:, i : i + 1]).transpose(1, 2) for i in range(2)], dim=1
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    changed = wave.clone()
    changed[:, 1] *= 2
    count = actual.shape[1] // 2
    torch.testing.assert_close(extractor(changed)[:, :count], actual[:, :count], atol=0, rtol=0)
    assert extractor.total_patches(128) == actual.shape[1]


def test_2d_positions_encode_same_times_and_distinct_ears():
    positions = get_binaural_sincos_pos_embed(16, 7)
    assert positions.shape == (14, 16)
    np.testing.assert_array_equal(positions[:7, :8], positions[7:, :8])
    np.testing.assert_array_equal(positions[:7, 8:], np.tile(positions[0, 8:], (7, 1)))
    np.testing.assert_array_equal(positions[7:, 8:], np.tile(positions[7, 8:], (7, 1)))
    assert not np.array_equal(positions[0, 8:], positions[7, 8:])
    # The published Nat channel indicator is zero for L, sin(0)/cos(0) for R.
    np.testing.assert_array_equal(positions[0], [0] * 4 + [1] * 4 + [0] * 8)
    np.testing.assert_array_equal(positions[7], [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
    model = WavJEPABackbone(nat_config())
    assert not model.model.pos_encoding_encoder.requires_grad
    times = model.token_timestamps_ms()
    count = times.numel() // 2
    torch.testing.assert_close(times[:count], times[count:], atol=0, rtol=0)


def test_mask_indices_repeat_complete_channel_sequences_without_cross_ear_leakage():
    model = WavJEPABackbone(nat_config())
    count = model.model.total_patches // 2
    masks = model.masker(4, count * 2, 2, generator=np.random.default_rng(24))
    for mask in masks:
        torch.testing.assert_close(mask[..., :count], mask[..., count:], atol=0, rtol=0)
    context, targets, combined = masks
    # If either ear's token is a target, neither ear at that time is visible
    # to the context encoder; each target block has the same pair of ears.
    assert not ((~context).unsqueeze(1) & targets).any()
    assert torch.equal(combined, context.unsqueeze(1) ^ targets)
    assert (targets[..., :count].sum(dim=-1) > 0).all()


def test_normalization_is_joint_preserving_interaural_level_ratio():
    model = WavJEPABackbone(nat_config())
    tone = torch.linspace(-1, 1, 128)
    wave = torch.stack((tone, 0.25 * tone)).reshape(1, 1, 2, 128)
    actual = model._normalized_windows(wave)
    assert actual.shape == (1, 2, 128)
    torch.testing.assert_close(actual[:, 1], actual[:, 0] * 0.25)
    torch.testing.assert_close(actual.mean(), torch.tensor(0.0), atol=1e-7, rtol=0)


def test_binaural_readout_and_masked_targets_use_same_normalized_teacher():
    model = WavJEPABackbone(nat_config()).train()
    wave = torch.randn(2, 3, 2, 128)
    tokens = model.encode_tokens(wave)
    assert tokens.shape == (2, 3, model.model.total_patches, 16)
    loss, targets = model.masked_loss_with_targets(wave, seed=17)
    torch.testing.assert_close(targets, model.encode_teacher_windows(wave), atol=0, rtol=0)
    assert targets.shape == (2, 3, 16) and not targets.requires_grad
    readout = model.encode_windows(wave, "normalized_top_k")
    assert readout.shape == (2, 3, 16) and readout.requires_grad
    loss.backward()
    for cnn in model.model.extract_audio.cnns:
        first = next(cnn.parameters())
        assert first.grad is not None and torch.isfinite(first.grad).all()
        assert first.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.model.teacher_encoder.parameters())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_binaural_compaction_matches_dense_loss_and_parameter_gradients(checkpointing):
    torch.manual_seed(30)
    dense = WavJEPABackbone(nat_config(use_gradient_checkpointing=checkpointing)).train()
    compact = deepcopy(dense)
    compact.model.compact_masked_tokens = True
    wave = torch.randn(2, 2, 2, 128)
    expected = dense.masked_loss(wave, seed=17)
    actual = compact.masked_loss(wave, seed=17)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    expected.backward()
    actual.backward()
    for (name, left), (other, right) in zip(
        dense.named_parameters(), compact.named_parameters(), strict=True
    ):
        assert name == other
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=1e-4, msg=name)


@pytest.mark.parametrize("channels", [1, 2])
def test_longer_masked_crops_share_learned_weights_and_do_not_change_window_positions(channels):
    model = WavJEPABackbone(nat_config(input_channels=channels, compact_masked_tokens=True))
    reference = WavJEPABackbone(
        nat_config(input_channels=channels, window_samples=256, compact_masked_tokens=True)
    )
    # A genuinely configured longer-window model is the numerical reference.
    state = reference.state_dict()
    for key, value in model.state_dict().items():
        if "pos_encoding_" not in key:
            state[key] = value
    reference.load_state_dict(state, strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    shape = (2, 8, 2, 256) if channels == 2 else (2, 8, 256)
    crops = torch.randn(shape)
    expected = reference.masked_loss(crops, seed=22)
    actual = model.masked_loss(crops, seed=22)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.backward()
    assert next(model.model.encoder.parameters()).grad is not None
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    assert model.window_samples == 128
    with pytest.raises(ValueError, match="expected waveform"):
        model.masked_loss_with_targets(crops, seed=22)
    with pytest.raises(ValueError, match="expected waveform"):
        model.encode_tokens(crops)


def test_binaural_checkpoint_roundtrip_and_no_implicit_mono_conversion(tmp_path):
    model = WavJEPABackbone(nat_config())
    path = tmp_path / "nat.pt"
    torch.save(model.model.state_dict(), path)
    restored = WavJEPABackbone(nat_config(), checkpoint_path=str(path))
    assert restored.construction_config()["input_channels"] == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)
    mono = WavJEPABackbone(nat_config(input_channels=1))
    torch.save(mono.model.state_dict(), path)
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    with pytest.raises(ValueError, match="incompatible"):
        restored.load_official_checkpoint(path)
    for key, value in before.items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_binaural_huggingface_layout_roundtrip_and_mono_rejection(tmp_path):
    from safetensors.torch import save_file

    model = WavJEPABackbone(nat_config())
    path = tmp_path / "model.safetensors"
    save_file({f"model.{k}": v for k, v in model.model.state_dict().items()}, path)
    restored = WavJEPABackbone(nat_config(), pretrained_path=str(path))
    wave = torch.randn(2, 2, 2, 128)
    torch.testing.assert_close(
        restored.masked_loss(wave, seed=1), model.masked_loss(wave, seed=1), atol=0, rtol=0
    )
    before = {k: v.clone() for k, v in restored.state_dict().items()}
    mono = WavJEPABackbone(nat_config(input_channels=1))
    save_file({f"model.{k}": v for k, v in mono.model.state_dict().items()}, path)
    with pytest.raises(ValueError):
        restored.load_huggingface_checkpoint(path)
    for key, value in before.items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_binaural_pretrained_flag_selects_nat_with_no_mono_fallback(monkeypatch):
    calls = []

    def selected_download(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("Nat artifact not cached")

    monkeypatch.setattr(
        "src.models.components.wavjepa_backbone.download_official_weights", selected_download
    )
    with torch.device("meta"), pytest.raises(FileNotFoundError, match="Nat artifact not cached"):
        WavJEPABackbone({"input_channels": 2}, pretrained=True, local_files_only=True)
    assert calls == [{"variant": "nat", "local_files_only": True}]


@pytest.mark.parametrize("channels", [0, 3, True, "2"])
def test_invalid_channel_config_is_rejected(channels):
    with pytest.raises(ValueError, match="input_channels"):
        WavJEPABackbone(nat_config(input_channels=channels))


@pytest.mark.parametrize("shape", [(2, 3, 128), (2, 3, 1, 128), (2, 3, 3, 128)])
def test_binaural_input_rejects_missing_or_wrong_channel_axis(shape):
    model = WavJEPABackbone(nat_config())
    with pytest.raises(ValueError, match="expected waveform"):
        model.masked_loss(torch.randn(shape), seed=17)


def test_binaural_masked_loss_is_fp32_with_finite_bfloat16_gradients():
    model = WavJEPABackbone(nat_config(compact_masked_tokens=True))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.masked_loss(torch.randn(2, 2, 2, 256), seed=7)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

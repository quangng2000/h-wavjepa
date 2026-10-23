"""Compare optional token packing against the original dense masked objective."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.models.components.wavjepa_backbone import WavJEPABackbone


def config(**overrides):
    return {
        "window_samples": 128,
        "conv_layers": [[8, 4, 2], [8, 3, 2]],
        "encoder_dim": 16,
        "encoder_heads": 2,
        "encoder_layers": 3,
        "decoder_dim": 8,
        "decoder_heads": 2,
        "decoder_layers": 2,
        "average_top_k_layers": 2,
        "masker": {
            "target_masks_per_context": 3,
            "context_mask_prob": 0.5,
            "context_mask_length": 2,
            "target_prob": 0.2,
            "target_length": 2,
        },
        **overrides,
    }


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("seed", [1, 17, 2026])
def test_compaction_preserves_masked_values_and_all_parameter_gradients(checkpointing, seed):
    torch.manual_seed(31)
    dense = WavJEPABackbone(config(use_gradient_checkpointing=checkpointing)).train()
    compact = WavJEPABackbone(
        config(use_gradient_checkpointing=checkpointing, compact_masked_tokens=True)
    ).train()
    compact.load_state_dict(dense.state_dict(), strict=True)
    waveform = torch.randn(3, 2, 128)
    masks = dense.masker(6, dense.model.total_patches, 1, generator=np.random.default_rng(seed))
    expected = dense.model(dense._normalized_windows(waveform), *masks)
    actual = compact.model(compact._normalized_windows(waveform), *masks)
    # Predictions outside the visible context/target union are intentionally unused.
    visible = ~masks[2].flatten(0, 1)
    torch.testing.assert_close(actual["preds"][visible], expected["preds"][visible])
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual["targets"], expected["targets"], rtol=0, atol=0)
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, left), (other, right) in zip(
        dense.named_parameters(), compact.named_parameters(), strict=True
    ):
        assert name == other
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=1e-4, msg=name)


def test_compaction_reduces_sequence_work_without_changing_dense_teacher_or_readouts():
    backbone = WavJEPABackbone(config(compact_masked_tokens=True)).train()
    lengths = {"encoder": [], "decoder": [], "teacher": []}
    handles = []
    for label, module in (
        ("encoder", backbone.model.encoder),
        ("decoder", backbone.model.decoder),
        ("teacher", backbone.model.teacher_encoder.layers[0]),
    ):
        handles.append(
            module.register_forward_pre_hook(
                lambda _, args, name=label: lengths[name].append(args[0].shape[1])
            )
        )
    try:
        waveform = torch.randn(3, 2, 128)
        loss, targets = backbone.masked_loss_with_targets(waveform, seed=17)
        assert loss.requires_grad and not targets.requires_grad
    finally:
        for handle in handles:
            handle.remove()
    full_length = backbone.model.total_patches
    assert lengths["teacher"] == [full_length]
    assert 0 < lengths["encoder"][0] < full_length
    assert 0 < lengths["decoder"][0] < full_length
    dense = WavJEPABackbone(config()).train()
    dense.load_state_dict(backbone.state_dict(), strict=True)
    for readout in ("legacy", "normalized_top_k"):
        torch.testing.assert_close(
            backbone.encode_windows(waveform, readout),
            dense.encode_windows(waveform, readout),
            rtol=0,
            atol=0,
        )


def test_compaction_state_dict_roundtrip_and_saved_config(tmp_path):
    compact = WavJEPABackbone(config(compact_masked_tokens=True))
    source = tmp_path / "official.pt"
    torch.save(compact.model.state_dict(), source)
    dense = WavJEPABackbone(config(), checkpoint_path=str(source))
    assert list(dense.state_dict()) == list(compact.state_dict())
    restored = WavJEPABackbone(compact.construction_config())
    assert restored.model.compact_masked_tokens
    restored.load_state_dict(dense.state_dict(), strict=True)


def test_compaction_rejects_unsupported_masks_and_training_dropout():
    backbone = WavJEPABackbone(config(compact_masked_tokens=True)).train()
    x = torch.randn(2, 5, 16)
    with pytest.raises(ValueError, match="boolean"):
        backbone.model.encoder_forward(x, torch.zeros(2, 5))
    with pytest.raises(ValueError, match="visible token"):
        backbone.model.encoder_forward(x, torch.ones(2, 5, dtype=torch.bool))
    backbone.model.encoder.layers[0].dropout.p = 0.1
    with pytest.raises(ValueError, match="zero dropout"):
        backbone.model.encoder_forward(x, torch.zeros(2, 5, dtype=torch.bool))


@pytest.mark.parametrize("value", [1, "true", None])
def test_compaction_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="compact_masked_tokens must be boolean"):
        WavJEPABackbone(config(compact_masked_tokens=value))


def test_compaction_mixed_precision_retains_float32_loss_and_finite_gradients():
    dense = WavJEPABackbone(config()).train()
    compact = deepcopy(dense)
    compact.model.compact_masked_tokens = True
    waveform = torch.randn(2, 2, 128)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = dense.masked_loss(waveform, seed=17)
        actual = compact.masked_loss(waveform, seed=17)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=5e-3)
    actual.backward()
    assert all(torch.isfinite(p.grad).all() for p in compact.parameters() if p.grad is not None)


@pytest.mark.parametrize("training", [False, True])
def test_compaction_preserves_uneven_rows_single_visible_and_unmasked_rows(training):
    dense = WavJEPABackbone(config()).train(training)
    compact = deepcopy(dense)
    compact.model.compact_masked_tokens = True
    x = torch.randn(3, 7, 16)
    mask = torch.tensor(
        [[False] * 7, [True, True, True, False, True, True, True], [False, True] * 3 + [False]]
    )
    with torch.set_grad_enabled(training):
        expected = dense.model.encoder_forward(x, mask)
        actual = compact.model.encoder_forward(x, mask)
    torch.testing.assert_close(actual[~mask], expected[~mask], rtol=1e-5, atol=1e-6)


def test_compaction_flag_is_compatible_with_official_pretrained_architecture(monkeypatch):
    def offline_download(**kwargs):
        raise FileNotFoundError("reached official checkpoint download")

    monkeypatch.setattr(
        "src.models.components.wavjepa_backbone.download_official_weights", offline_download
    )
    with pytest.raises(FileNotFoundError, match="reached official checkpoint download"):
        WavJEPABackbone({"compact_masked_tokens": True}, pretrained=True)

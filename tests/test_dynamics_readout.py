"""Regression checks for aligned forecasting readouts and teacher reuse."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from test_hierarchical_dynamics import checkpoint_for, model_factory, waveform

from src.dynamics_inference import load_model
from src.models.components.wavjepa_backbone import WavJEPABackbone


def multilayer_model(**overrides):
    config = model_factory().backbone.construction_config()
    config.update(encoder_layers=3, average_top_k_layers=2)
    return model_factory(
        backbone=WavJEPABackbone(config=config),
        readout_mode="normalized_top_k",
        **overrides,
    )


def test_matching_multilayer_readouts_align_all_initial_online_and_ema_levels():
    model = multilayer_model().eval()
    audio = waveform()
    online = model.encode(audio)
    teacher = model.encode_targets(audio)
    for student, target in zip(online, teacher, strict=True):
        torch.testing.assert_close(student, target, rtol=1e-5, atol=1e-6)
        assert student.requires_grad
        assert not target.requires_grad
    online[-1].square().mean().backward()
    assert any(p.grad is not None for p in model.backbone.model.encoder.parameters())
    assert all(p.grad is None for p in model.teacher_abstractions.parameters())


@pytest.mark.parametrize("masked_weight, teacher_window_count", [(1.0, 9), (0.0, 3)])
def test_forecasting_never_repeats_teacher_windows(masked_weight, teacher_window_count):
    model = multilayer_model(masked_weight=masked_weight).eval()
    audio = waveform()
    with patch.object(
        model.backbone.model, "_forward_teacher", wraps=model.backbone.model._forward_teacher
    ) as teacher:
        result = model.compute_loss(audio, seed=31)
    assert teacher.call_count == 1
    assert teacher.call_args.args[0].shape[0] == audio.shape[0] * teacher_window_count
    reference = model.encode_targets(audio)
    for level, horizon in enumerate(model.horizons):
        torch.testing.assert_close(
            result["targets"][level], reference[level][:, model.context_windows - 1 + horizon]
        )


def test_reused_teacher_preserves_full_loss_and_parameter_gradients():
    cached = multilayer_model().train()
    recomputed = deepcopy(cached)
    audio = waveform()
    original = recomputed.backbone.masked_loss_with_targets

    def discard_targets(*args, **kwargs):
        loss, _ = original(*args, **kwargs)
        return loss, None

    expected = cached.compute_loss(audio, seed=43)
    with patch.object(recomputed.backbone, "masked_loss_with_targets", discard_targets):
        actual = recomputed.compute_loss(audio, seed=43)
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, left), (_, right) in zip(
        cached.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)


def test_checkpoint_persists_readout_and_rejects_silent_semantic_change(tmp_path):
    source = multilayer_model().eval()
    checkpoint = checkpoint_for(source)
    path = tmp_path / "aligned.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.readout_mode == "normalized_top_k"
    legacy = model_factory(backbone=WavJEPABackbone(config=source.backbone.config))
    with pytest.raises(ValueError, match="configuration mismatch"):
        legacy.on_load_checkpoint(checkpoint)
    audio = waveform()
    for left, right in zip(source.encode(audio), restored.encode(audio), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_checkpoint_without_readout_field_retains_legacy_semantics(tmp_path):
    legacy = model_factory().eval()
    checkpoint = checkpoint_for(legacy)
    checkpoint[legacy.CHECKPOINT_KEY]["dynamics"].pop("readout_mode")
    path = tmp_path / "legacy-readout.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.readout_mode == "legacy"
    audio = waveform()
    for left, right in zip(legacy.encode(audio), restored.encode(audio), strict=True):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

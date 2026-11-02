"""Learned aggregation is included in EMA and the exact checkpoint contract."""

import pytest
import torch
from test_hierarchical_dynamics import checkpoint_for, model_factory, waveform

from src.dynamics_inference import load_model


def conv_model(**overrides):
    return model_factory(
        **({"aggregation_mode": "conv1d", "aggregation_windows": [1, 2, 2]} | overrides)
    )


def assert_same_state(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, tensor in left.state_dict().items():
        torch.testing.assert_close(tensor, right.state_dict()[name], rtol=0, atol=0)


def test_ema_tracks_convolution_weights_after_an_online_change():
    model = conv_model(ema_decay=0.75)
    online = next(model.abstractions[0][0].parameters())
    teacher = next(model.teacher_abstractions[0][0].parameters())
    before = teacher.detach().clone()
    with torch.no_grad():
        online.add_(2)
    model.update_teacher()
    torch.testing.assert_close(teacher, before + 0.5)
    assert model.ema_updates == 1
    assert not model.teacher_abstractions.training


def test_checkpoint_roundtrip_preserves_learned_convolutions_and_predictions(tmp_path):
    model = conv_model().eval().requires_grad_(False)
    with torch.no_grad():
        next(model.abstractions[0][0].parameters()).add_(0.02)
    model.update_teacher()
    checkpoint = checkpoint_for(model)
    assert checkpoint[model.CHECKPOINT_KEY]["dynamics"]["aggregation_mode"] == "conv1d"
    path = tmp_path / "convolution.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.config["aggregation_mode"] == "conv1d"
    assert_same_state(model, restored)
    audio = waveform()
    # Match both inference mode and parameter flags: the loader freezes its weights.
    with torch.no_grad():
        for left, right in zip(model.encode(audio), restored.encode(audio), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(
        model.encode_targets(audio), restored.encode_targets(audio), strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("windows", [[1, 1, 1], [1, 2, 2]])
def test_old_checkpoint_without_mode_retains_its_original_mean_behavior(tmp_path, windows):
    model = model_factory(aggregation_windows=windows).eval().requires_grad_(False)
    checkpoint = checkpoint_for(model)
    checkpoint[model.CHECKPOINT_KEY]["dynamics"].pop("aggregation_mode")
    path = tmp_path / "old.ckpt"
    torch.save(checkpoint, path)
    restored = load_model(path)
    assert restored.config["aggregation_mode"] == "mean"
    assert_same_state(model, restored)
    audio = waveform()
    with torch.no_grad():
        for left, right in zip(model.encode(audio), restored.encode(audio), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_resume_rejects_a_change_between_mean_and_convolution():
    mean = model_factory(aggregation_windows=[1, 2, 2])
    convolution = conv_model()
    with pytest.raises(ValueError, match="configuration mismatch"):
        convolution.on_load_checkpoint(checkpoint_for(mean))
    with pytest.raises(ValueError, match="configuration mismatch"):
        mean.on_load_checkpoint(checkpoint_for(convolution))

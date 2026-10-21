"""Local shape, information-flow, objective, and teacher contracts."""

from functools import partial

import pytest
import torch

from src.models.components.hierarchy import HierarchyEncoder, patchify
from src.models.components.masking import grouped_patch_mask
from src.models.hwavjepa_module import HWavJEPAModule
from src.objectives.losses import reference_reconstruction_loss


def make_model(protection_weight: float = 1.0, flat: bool = False) -> HWavJEPAModule:
    return HWavJEPAModule(
        encoder=HierarchyEncoder(
            patch_size=8,
            widths=(8,) if flat else (8, 12, 16),
            pool_factors=() if flat else (2, 2),
        ),
        optimizer=partial(torch.optim.SGD, lr=0.05),
        protection_weight=protection_weight,
        protected_dim=8,
    )


def test_shapes_and_unchanged_protected_readout() -> None:
    model = make_model()
    waveform = torch.randn(2, 128)
    result = model.encode(waveform)
    assert [tuple(level.shape) for level in result["levels"]] == [
        (2, 16, 8),
        (2, 8, 12),
        (2, 4, 16),
    ]
    assert result["protected"].shape == (2, 16, 8)
    assert all(readout["protected"] is result["protected"] for readout in result["readouts"])
    assert torch.equal(model.encode(waveform[:, None, :])["levels"][0], result["levels"][0])


@pytest.mark.parametrize("flat", [False, True])
def test_masked_predictions_do_not_see_hidden_samples(flat: bool) -> None:
    torch.manual_seed(12)
    model = make_model(flat=flat).eval()
    waveform = torch.randn(2, 128)
    mask = model.make_patch_mask(waveform)
    altered = patchify(waveform.clone(), 8)
    altered[mask] += 100 * torch.randn_like(altered[mask])
    before = model.predictive_outputs(waveform, mask)
    after = model.predictive_outputs(altered.reshape(2, -1), mask)
    for first, second in zip(before["predictions"], after["predictions"], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(before["targets"], after["targets"], strict=True)
    )


def test_teacher_has_no_grad_and_updates_only_on_optimizer_step() -> None:
    model = make_model().train()
    assert not model.teacher.training
    optimizer = model.configure_optimizers()
    original = [parameter.clone() for parameter in model.teacher.parameters()]
    waveform = torch.randn(2, 128)
    loss = model.compute_loss(waveform)["loss"]
    loss.backward()
    assert model.ema_updates.item() == 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(
        torch.equal(old, new) for old, new in zip(original, model.teacher.parameters(), strict=True)
    )
    optimizer.step()
    assert model.ema_updates.item() == 1
    for old, teacher, online in zip(
        original, model.teacher.parameters(), model.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(teacher, model.ema_decay * old + (1 - model.ema_decay) * online)


def test_reference_is_raw_and_not_an_ema_or_trainable_target() -> None:
    raw = torch.randn(2, 4, 8, requires_grad=True)
    prediction = torch.randn_like(raw, requires_grad=True)
    reference_reconstruction_loss(prediction, raw).backward()
    assert raw.grad is None
    assert prediction.grad is not None
    model = make_model()
    waveform = torch.randn(2, 128)
    original = waveform.clone()
    patches = patchify(waveform, 8)
    result = model.compute_loss(waveform)
    reconstruction = model.protected_decoder(model.protected_encoder(patches))
    torch.testing.assert_close(
        result["protection_loss"], (reconstruction - patches).square().mean()
    )
    result["loss"].backward()
    model.configure_optimizers().step()
    assert torch.equal(waveform, original)


def test_unprotected_variant_has_no_unused_protection_parameters() -> None:
    model = make_model(protection_weight=0)
    assert model.protected_encoder is None and model.protected_decoder is None
    assert not any(name.startswith("protected_") for name, _ in model.named_parameters())
    waveform = torch.randn(2, 128)
    result = model.compute_loss(waveform)
    assert result["protection_loss"].item() == 0
    torch.testing.assert_close(result["loss"], result["prediction_loss"])
    assert model.encode(waveform)["protected"] is None


def test_mask_group_contract_and_validation() -> None:
    torch.manual_seed(2)
    mask = grouped_patch_mask(3, 16, 4, 0.5, torch.device("cpu"))
    assert torch.equal(mask.reshape(3, 4, 4), mask[:, ::4].unsqueeze(-1).expand(3, 4, 4))
    assert torch.all(mask.sum(dim=1) == 8)
    with pytest.raises(ValueError, match="two groups"):
        grouped_patch_mask(1, 4, 4, 0.5, torch.device("cpu"))
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.randn(2, 127), 8)
    model = make_model()
    with pytest.raises(ValueError, match="visible and hidden"):
        model.predictive_outputs(torch.randn(2, 128), torch.ones(2, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match="multiple"):
        HWavJEPAModule(model.encoder, partial(torch.optim.SGD, lr=0.1), mask_group_size=2)


def test_partial_hidden_support_propagates_to_every_ancestor() -> None:
    model = make_model().eval()
    waveform = torch.randn(2, 128)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, 1] = True
    outputs = model.predictive_outputs(waveform, mask)
    assert all(level_mask.sum(dim=1).tolist() == [1, 1] for level_mask in outputs["masks"])
    assert outputs["masks"][1][:, 0].all()
    assert outputs["masks"][2][:, 0].all()
    changed = waveform.clone()
    changed[:, 8:16] = torch.randn(2, 8) * 100
    changed_outputs = model.predictive_outputs(changed, mask)
    for before, after in zip(outputs["predictions"], changed_outputs["predictions"], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_full_input_protected_branch_cannot_feed_masked_predictors() -> None:
    model = make_model().eval()
    waveform = torch.randn(2, 128)
    mask = model.make_patch_mask(waveform)
    before = model.predictive_outputs(waveform, mask)["predictions"]
    with torch.no_grad():
        for parameter in model.protected_encoder.parameters():
            parameter.fill_(123.0)
        for parameter in model.protected_decoder.parameters():
            parameter.fill_(-456.0)
    after = model.predictive_outputs(waveform, mask)["predictions"]
    for first, second in zip(before, after, strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

"""Compare the adapted regularizer with pinned, unmodified H-JEPA source."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.objectives.sigreg import SIGReg


def _reference():
    directory = Path(__file__).parent / "fixtures" / "h_jepa"
    manifest = json.loads((directory / "UPSTREAM.json").read_text())
    source = directory / "loss.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"]
    spec = importlib.util.spec_from_file_location("pinned_h_jepa_loss", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SIGReg


@pytest.mark.parametrize("shape", [(1, 2, 16), (5, 4, 32)])
@pytest.mark.parametrize("projections", [8, 128])
def test_sigreg_value_and_gradient_match_independent_upstream(shape, projections):
    reference = _reference()(num_proj=projections)
    adapted = SIGReg(num_proj=projections)
    original = torch.randn(shape, requires_grad=True)
    local = original.detach().clone().requires_grad_()
    torch.manual_seed(1234)
    expected = reference(original)
    expected.backward()
    torch.manual_seed(1234)
    actual = adapted(local)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(local.grad, original.grad, atol=0, rtol=0)

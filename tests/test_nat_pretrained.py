"""Pinned official Nat export layout, fixed positions and safe download contract."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src import pretrained
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.vendor.wavjepa.pos_embed import get_binaural_sincos_pos_embed


def test_nat_download_requests_only_immutable_digest_checked_tensor_artifact(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"Nat download contract fixture")
    monkeypatch.setattr(pretrained, "NAT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    assert (
        pretrained.download_official_weights(
            variant="nat", local_files_only=True, cache_dir=tmp_path
        )
        == path
    )
    assert calls == [
        {
            "repo_id": "labhamlet/wavjepa-nat-base",
            "filename": "model.safetensors",
            "revision": "15d95ff67fa98117b17e83a1653bbca97877ff6f",
            "local_files_only": True,
            "cache_dir": tmp_path,
        }
    ]


def test_nat_download_rejects_corruption_and_unknown_variant(tmp_path, monkeypatch):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"corrupted")
    monkeypatch.setattr(pretrained, "hf_hub_download", lambda **kwargs: str(path))
    with pytest.raises(ValueError, match="SHA-256"):
        pretrained.download_official_weights(variant="nat")
    with pytest.raises(ValueError, match="variant"):
        pretrained.download_official_weights(variant="naturalistic")
    with pytest.raises(ValueError, match="variant"):
        WavJEPABackbone.from_pretrained(variant="naturalistic")


def test_full_nat_model_layout_matches_published_safetensors_header_without_weight_download():
    fixture = json.loads((Path(__file__).parent / "fixtures/wavjepa_nat_header.json").read_text())
    assert fixture["revision"] == pretrained.NAT_REVISION
    assert fixture["sha256"] == pretrained.NAT_SHA256
    with torch.device("meta"):
        backbone = WavJEPABackbone({"input_channels": 2})
    state = {f"model.{key}": value for key, value in backbone.model.state_dict().items()}
    assert len(state) == 465 and state.keys() == fixture["tensors"].keys()
    for key, value in state.items():
        assert list(value.shape) == fixture["tensors"][key]["shape"], key
        assert value.dtype == torch.float32 and fixture["tensors"][key]["dtype"] == "F32"
    assert backbone.model.total_patches == 400


@pytest.mark.parametrize(
    "width,digest",
    [
        (768, "42e124cb8ef2cc8d5801f2e9db89983ee77b4c3578c172d537be35ac318ecb14"),
        (384, "2d91787848963ba37571804c5556ffe1aa9028493ec535ecd184ba96a314581e"),
    ],
)
def test_independent_nat_positions_match_actual_published_tensor_bytes(width, digest):
    # Golden hashes from HTTP ranges of the pinned safetensors file, not from
    # executing any Hub Python or regenerating the expected values locally.
    values = get_binaural_sincos_pos_embed(width, 200).astype("<f4")
    assert hashlib.sha256(values.tobytes()).hexdigest() == digest


def test_nat_position_adaptation_only_changes_fixed_tensors_and_rejects_mono_positions():
    with torch.device("meta"):
        backbone = WavJEPABackbone({"input_channels": 2, "window_samples": 16160})
    state = dict(backbone.model.state_dict())
    for key, width in (("pos_encoding_encoder", 768), ("pos_encoding_decoder", 384)):
        state[key] = torch.from_numpy(get_binaural_sincos_pos_embed(width, 200)).float()[None]
    adapted = backbone._adapt_fixed_positions(state)
    for key, value in state.items():
        if key.startswith("pos_encoding_"):
            assert adapted[key].shape[1] == 200
        else:
            assert adapted[key] is value
    changed = dict(state)
    changed["pos_encoding_encoder"] = state["pos_encoding_encoder"].clone()
    changed["pos_encoding_encoder"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="exact official"):
        backbone._adapt_fixed_positions(changed)
    mono = dict(state)
    mono["pos_encoding_encoder"] = state["pos_encoding_encoder"][:, :200]
    # Use a differently sized destination so adaptation is actually requested.
    with torch.device("meta"):
        other = WavJEPABackbone({"input_channels": 2, "window_samples": 24160})
    with pytest.raises(ValueError, match="exact official"):
        other._adapt_fixed_positions(mono)

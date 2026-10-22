"""Pinned download contract; network and large artifacts are deliberately optional."""

import hashlib

import pytest

from src import pretrained


def test_download_only_requests_pinned_tensor_artifact(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"download-contract-fixture")
    monkeypatch.setattr(
        pretrained, "OFFICIAL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    assert pretrained.download_official_weights(local_files_only=True, cache_dir=tmp_path) == path
    assert calls == [
        {
            "repo_id": "labhamlet/wavjepa-base",
            "filename": "model.safetensors",
            "revision": "6be4a5093f6c9adbbd55e00b6e5b8f067aa03345",
            "local_files_only": True,
            "cache_dir": tmp_path,
        }
    ]


def test_download_rejects_corrupt_artifact(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"incorrect artifact")
    monkeypatch.setattr(pretrained, "hf_hub_download", lambda **kwargs: str(path))
    with pytest.raises(ValueError, match="SHA-256"):
        pretrained.download_official_weights()

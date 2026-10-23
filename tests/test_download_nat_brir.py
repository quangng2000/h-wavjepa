"""The accelerated downloader remains pinned and confined to one scratch tree."""

import json
import os
from pathlib import Path

import pytest

from src import download_nat_brir
from src.prepare_nat_assets import BRIR_REPOSITORY, BRIR_REVISION

SOURCE = "train/webdataset_output/shard-000069.tar"


def test_hub_download_is_pinned_anonymous_and_uses_only_owned_cache(tmp_path, monkeypatch):
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_XET_CACHE",
        "HF_XET_LOG_DEST",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES",
        "HF_XET_LOG_DIR_MAX_SIZE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_VERBOSITY",
        "HF_DEBUG",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ETAG_TIMEOUT",
        "HF_HUB_DOWNLOAD_TIMEOUT",
    ):
        monkeypatch.setenv(name, "previous-value")
    options = []

    def download(**kwargs):
        options.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"tiny fixture; parent verifies publisher SHA and actual size")
        return str(path)

    target = tmp_path / "source"
    path = download_nat_brir.download_brir(SOURCE, target, downloader=download)
    assert path == target / SOURCE
    assert options == [
        {
            "repo_id": BRIR_REPOSITORY,
            "repo_type": "dataset",
            "revision": BRIR_REVISION,
            "filename": SOURCE,
            "local_dir": str(target),
            "cache_dir": str(target / ".hf-home/hub"),
            "token": False,
            "etag_timeout": 30,
        }
    ]
    assert os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"
    assert os.environ["HF_DEBUG"] == "0"
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        assert Path(os.environ[key]).is_relative_to(target)


@pytest.mark.parametrize(
    "source",
    [
        "train/webdataset_output/shard-000070.tar",
        "eval/webdataset_output/shard-000000.tar",
        "../archive.tar",
        "https://example.com/source.tar",
        "/train/webdataset_output/shard-000000.tar",
    ],
)
def test_other_sources_are_rejected_before_downloader(source, tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        download_nat_brir.download_brir(
            source, tmp_path, downloader=lambda **_: pytest.fail("network")
        )


def test_shared_cache_or_symlink_return_is_rejected(tmp_path, monkeypatch):
    # Restore environment changes after this process-local helper is exercised.
    monkeypatch.setattr(download_nat_brir.os, "environ", dict(os.environ))
    outside = tmp_path / "outside.tar"
    outside.write_bytes(b"other")
    with pytest.raises(ValueError, match="single-file"):
        download_nat_brir.download_brir(SOURCE, tmp_path / "work", downloader=lambda **_: outside)

    def symlink(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.symlink_to(outside)
        return path

    with pytest.raises(ValueError, match="single-file"):
        download_nat_brir.download_brir(SOURCE, tmp_path / "symlink-work", downloader=symlink)


def test_cli_reports_only_the_local_result_path(tmp_path, monkeypatch, capsys):
    target = tmp_path / SOURCE
    monkeypatch.setattr(download_nat_brir, "download_brir", lambda *_: target)
    download_nat_brir.main(["--source-path", SOURCE, "--local-dir", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {"path": str(target)}

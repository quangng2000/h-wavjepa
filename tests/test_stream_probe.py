"""Bounded probe proves real production decode and cleanup using staged S3 fixtures."""

import hashlib
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("webdataset")
pytest.importorskip("boto3")

from scripts.probe_sharded_audio import ProbeLimit, deadline, run_probe, select_shards
from src.data import sharded_audio_datamodule as production


def fixture_index(tmp_path, monkeypatch):
    from test_stage_audioset import FakeS3, runner, source_archive

    store = FakeS3()
    source = source_archive(tmp_path)
    index = runner(tmp_path, store, [source]).run(metadata=False)
    location = tmp_path / "frozen-index.json"
    location.write_text(json.dumps(index))
    monkeypatch.setattr(production, "_s3_client", lambda _: store)
    monkeypatch.setattr(
        production.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3)
    )
    return store, index, location


def test_probe_uses_production_stream_records_checksum_and_removes_own_cache(tmp_path, monkeypatch):
    store, index, location = fixture_index(tmp_path, monkeypatch)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unrelated data")
    before = len(store.gets)
    result = run_probe(
        index_path=str(location),
        endpoint=index["endpoint"],
        cache_parent=tmp_path,
        allow_partial=True,
    )
    assert result["status"] == "completed"
    assert result["frozen_index_sha256"] == hashlib.sha256(location.read_bytes()).hexdigest()
    assert result["selected_source_count"] == 1
    assert result["selected_payload_bytes"] == index["shards"][0]["bytes"]
    assert result["measurements"]["clips"] == 1
    assert result["measurements"]["s3_get_object_calls"] == len(store.gets) - before == 1
    assert result["measurements"]["s3_body_bytes"] == index["shards"][0]["bytes"]
    assert result["measurements"]["pcm_decode_seconds"] > 0
    assert result["measurements"]["cold_clips_per_second"] > 0
    assert [
        row["required_clips_per_second"] for row in result["required_aggregate_input_rates"]
    ] == [134.4, 211.2]
    assert result["owned_temporary_cache_removed"]
    assert not list(tmp_path.glob("hwavjepa-read-probe-*"))
    assert sentinel.read_text() == "unrelated data"


def test_probe_fails_closed_on_corrupt_remote_shard_and_cleans_up(tmp_path, monkeypatch):
    store, index, location = fixture_index(tmp_path, monkeypatch)
    key = index["shards"][0]["key"]
    store.objects[key] = b"x" + store.objects[key][1:]
    report = {}
    with pytest.raises(ValueError, match="SHA256"):
        run_probe(
            index_path=str(location),
            endpoint=index["endpoint"],
            cache_parent=tmp_path,
            allow_partial=True,
            report=report,
        )
    assert report["owned_temporary_cache_removed"]
    assert not list(tmp_path.glob("hwavjepa-read-probe-*"))


def test_probe_requires_explicit_partial_index_permission(tmp_path, monkeypatch):
    _, index, location = fixture_index(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires a complete"):
        run_probe(index_path=str(location), endpoint=index["endpoint"], cache_parent=tmp_path)
    assert not list(tmp_path.glob("hwavjepa-read-probe-*"))


def test_probe_selection_enforces_byte_and_source_limits():
    shards = [{"bytes": 700 * 1024**2}, {"bytes": 700 * 1024**2}, {"bytes": 50 * 1024**2}]
    assert select_shards(shards, 2, 1500 * 1024**2) == shards[:2]
    assert select_shards(shards, 2, 750 * 1024**2) == [shards[0], shards[2]]
    with pytest.raises(ValueError, match="1 or 2"):
        select_shards(shards, 3, 1500 * 1024**2)
    with pytest.raises(ValueError, match="at most 1.5"):
        select_shards(shards, 2, 2 * 1024**3)
    with pytest.raises(ValueError, match="No training shard"):
        select_shards(shards, 2, 1024)


def test_deadline_escapes_sdk_exception_retry_handlers_and_cleans_cache(tmp_path, monkeypatch):
    store, index, location = fixture_index(tmp_path, monkeypatch)
    attempts = []
    swallowed = []

    def blocked_get(**kwargs):
        attempts.append(True)
        try:
            time.sleep(1)
        except Exception:
            # botocore's HTTP adapter catches Exception and may wrap/retry it.
            swallowed.append(True)
            raise AssertionError("Hard cancellation must bypass retry wrappers")
        raise AssertionError("The deadline must interrupt the blocked operation")

    monkeypatch.setattr(store, "get_object", blocked_get)
    report = {}
    with pytest.raises(ProbeLimit), deadline(0.05):
        run_probe(
            index_path=str(location),
            endpoint=index["endpoint"],
            cache_parent=tmp_path,
            allow_partial=True,
            report=report,
        )
    assert attempts == [True] and not swallowed
    assert report["owned_temporary_cache_removed"]
    assert not list(tmp_path.glob("hwavjepa-read-probe-*"))

"""Rolling private-S3 staging using real tiny preparation archives and a fake S3."""

import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import pytest

from src.prepare_audioset import REVISION, Source, connect_ledger, prepare_source, sha256_file
from src.stage_audioset import Stager


class MissingObject(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.gets = []
        self.fail_tar_upload = False
        self.fail_index_upload = False
        self.corrupt_tar_readback = False

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        return {
            "Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)][:MaxKeys]
        }

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        payload = self.objects[Key]
        return {
            "ContentLength": len(payload),
            "ETag": hashlib.sha256(payload).hexdigest(),
            "Metadata": {},
        }

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise MissingObject()
        self.gets.append(Key)
        payload = self.objects[Key]
        if Key.endswith(".tar") and self.corrupt_tar_readback:
            payload = b"x" + payload[1:]
        return {"Body": io.BytesIO(payload), **self.head_object(Bucket=Bucket, Key=Key)}

    def put_object(self, *, Bucket, Key, Body, ContentLength, **kwargs):
        if (Key.endswith(".tar") and self.fail_tar_upload) or (
            Key.endswith("index.json") and self.fail_index_upload
        ):
            raise OSError("simulated upload failure")
        if isinstance(Body, bytes):
            payload = Body
        else:
            payload = b"".join(iter(lambda: Body.read(16_384), b""))
        assert len(payload) == ContentLength
        self.objects[Key] = payload
        self.puts.append(Key)
        return {"ETag": hashlib.sha256(payload).hexdigest()}


def source_archive(
    tmp_path, name="unbal_train000", split="train", directory="unbal_train", video_id="aaaaaaaaaaa"
):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x01\x00" * 160_000)
    path = tmp_path / f"{name}.tar"
    with tarfile.open(path, "w") as handle:
        member = tarfile.TarInfo(f"audio/{directory}/{video_id}.wav")
        member.size = len(audio.getvalue())
        handle.addfile(member, io.BytesIO(audio.getvalue()))
    return Source(name, str(path), sha256_file(path), path.stat().st_size, split, directory)


def fixture_prepare(source, folder):
    with connect_ledger(folder) as db:
        prepare_source(source, folder, db, workers=1, minimum_free_bytes=0)


def runner(tmp_path, store, sources, **kwargs):
    catalog = tmp_path / "catalog.json"
    if not catalog.exists():
        catalog.write_text(json.dumps({"revision": REVISION, "fixture": True}))
    return Stager(
        tmp_path / "stage",
        store,
        bucket="private-bucket",
        endpoint="https://s3.example.invalid",
        sources=sources,
        catalog=catalog,
        ffmpeg_version="fixture FFmpeg",
        prepare_callback=fixture_prepare,
        **kwargs,
    )


def test_success_verifies_remote_and_purges_only_owned_source_work(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    result = runner(tmp_path, store, [source]).run(metadata=False)
    assert result["status"] == "partial"
    assert result["full_cohort_sources"] == 889
    assert result["verified_sources"] == 1
    assert result["split_counts"]["train"] == 1
    assert not (tmp_path / "stage" / "work" / source.name).exists()
    shard = result["shards"][0]
    assert hashlib.sha256(store.objects[shard["key"]]).hexdigest() == shard["sha256"]
    assert shard["sha256"] in shard["key"]
    assert shard["key"] in store.gets
    with tarfile.open(fileobj=io.BytesIO(store.objects[shard["key"]]), mode="r:") as archive:
        names = archive.getnames()
        assert "train.jsonl" in names
        assert "clips.provenance.jsonl" in names
        assert any(name.endswith("aaaaaaaaaaa.wav") for name in names)
        manifest = json.loads(archive.extractfile("train.jsonl").read())
        assert manifest["recording_id"] == "aaaaaaaaaaa"
    assert json.loads(store.objects[result["index_key"]])["verified_sources"] == 1


def test_failed_upload_keeps_wavs_then_resume_uploads_and_cleans(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    store.fail_tar_upload = True
    with pytest.raises(OSError, match="upload failure"):
        runner(tmp_path, store, [source]).run(metadata=False)
    folder = tmp_path / "stage" / "work" / source.name
    assert list(folder.rglob("*.wav"))
    assert (folder / "prepared.tar").exists()
    store.fail_tar_upload = False
    result = runner(tmp_path, store, [source]).run(metadata=False)
    assert result["verified_sources"] == 1
    assert not folder.exists()


def test_successful_resume_skips_preparation_and_full_readback_for_unchanged_object(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    first = runner(tmp_path, store, [source]).run(metadata=False)
    key = first["shards"][0]["key"]
    initial_reads = store.gets.count(key)
    initial_puts = store.puts.count(key)
    resumed = runner(tmp_path, store, [source])
    resumed.prepare_callback = lambda *args: pytest.fail(
        "Already verified source must not be prepared again"
    )
    second = resumed.run(metadata=False)
    assert second["split_counts"] == first["split_counts"]
    assert store.puts.count(key) == initial_puts
    assert store.gets.count(key) == initial_reads


def test_cross_split_id_collision_remains_fatal_after_prior_work_evicted(tmp_path):
    train = source_archive(tmp_path)
    evaluation = source_archive(tmp_path, name="eval00", split="test", directory="eval")
    store = FakeS3()
    sources = [train, evaluation]
    first = runner(tmp_path, store, sources).run([train.name], metadata=False)
    assert first["split_counts"]["train"] == 1
    for _ in range(2):
        with pytest.raises(ValueError, match="Cross-split"):
            runner(tmp_path, store, sources).run([evaluation.name], metadata=False)
    assert (tmp_path / "stage" / "work" / evaluation.name).exists()
    assert sum(key.endswith(".tar") for key in store.objects) == 1


def test_corrupt_remote_readback_keeps_local_audio_and_does_not_enroll(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    store.corrupt_tar_readback = True
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner(tmp_path, store, [source]).run(metadata=False)
    assert list((tmp_path / "stage" / "work" / source.name).rglob("*.wav"))
    store.corrupt_tar_readback = False
    assert runner(tmp_path, store, [source]).run(metadata=False)["verified_sources"] == 1


def test_crash_after_remote_verification_before_enrollment_recovers_once(tmp_path, monkeypatch):
    source = source_archive(tmp_path)
    store = FakeS3()
    first = runner(tmp_path, store, [source])

    def crash(*args):
        raise RuntimeError("crash before durable enrollment")

    monkeypatch.setattr(first, "_enroll", crash)
    with pytest.raises(RuntimeError, match="before durable"):
        first.run(metadata=False)
    key = next(key for key in store.objects if key.endswith(".tar"))
    assert (tmp_path / "stage" / "work" / source.name).exists()
    result = runner(tmp_path, store, [source]).run(metadata=False)
    assert result["verified_sources"] == 1
    assert store.puts.count(key) == 1
    assert store.gets.count(key) == 2


def test_failed_index_upload_keeps_local_work_until_resume_publishes_index(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    store.fail_index_upload = True
    with pytest.raises(OSError, match="upload failure"):
        runner(tmp_path, store, [source]).run(metadata=False)
    assert (tmp_path / "stage" / "work" / source.name).exists()
    store.fail_index_upload = False
    result = runner(tmp_path, store, [source]).run(metadata=False)
    assert result["verified_sources"] == 1
    assert not (tmp_path / "stage" / "work" / source.name).exists()


def test_missing_local_registry_with_populated_prefix_fails_closed(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    store.objects["audioset/previous-cohort/something"] = b"existing"
    with pytest.raises(ValueError, match="local global ledger is absent"):
        runner(tmp_path, store, [source]).run(metadata=False)
    assert not any(key.endswith(".tar") for key in store.objects)


def test_changed_remote_object_after_success_is_detected_on_resume(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    first = runner(tmp_path, store, [source]).run(metadata=False)
    key = first["shards"][0]["key"]
    payload = bytearray(store.objects[key])
    payload[100] ^= 1
    store.objects[key] = bytes(payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner(tmp_path, store, [source]).run(metadata=False)


def test_upload_budget_retains_prepared_work_and_marks_partial(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    result = runner(tmp_path, store, [source], max_upload_bytes=1000).run(metadata=False)
    assert result["status"] == "partial"
    assert result["verified_sources"] == 0
    assert "budget" in result["stop_reason"]
    assert list((tmp_path / "stage" / "work" / source.name).rglob("*.wav"))
    assert not any(key.endswith(".tar") for key in store.objects)


def test_source_limit_preserves_partial_index(tmp_path):
    first = source_archive(tmp_path)
    second = source_archive(tmp_path, name="unbal_train001", video_id="bbbbbbbbbbb")
    store = FakeS3()
    result = runner(tmp_path, store, [first, second], max_sources=1).run(metadata=False)
    assert result["verified_sources"] == 1
    assert result["status"] == "partial"
    assert "source limit" in result["stop_reason"]


def test_refuses_unowned_work_directory_without_deleting_it(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    folder = tmp_path / "stage" / "work" / source.name
    folder.mkdir(parents=True)
    (folder / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="unowned"):
        runner(tmp_path, store, [source]).run(metadata=False)
    assert (folder / "precious.txt").read_text() == "keep"


def test_truncated_global_ledger_cannot_silently_forget_staged_ids(tmp_path):
    source = source_archive(tmp_path)
    store = FakeS3()
    runner(tmp_path, store, [source]).run(metadata=False)
    (tmp_path / "stage" / "staging.sqlite3").write_bytes(b"")
    with pytest.raises(ValueError, match="ledger missing or empty"):
        runner(tmp_path, store, [source]).run(metadata=False)


def test_restored_stale_ledger_cannot_replace_newer_remote_index(tmp_path):
    train = source_archive(tmp_path)
    evaluation = source_archive(tmp_path, name="eval00", split="test", directory="eval")
    sources = [train, evaluation]
    store = FakeS3()
    store.fail_tar_upload = True
    with pytest.raises(OSError):
        runner(tmp_path, store, sources).run([train.name], metadata=False)
    ledger = tmp_path / "stage" / "staging.sqlite3"
    old_state = ledger.read_bytes()
    store.fail_tar_upload = False
    runner(tmp_path, store, sources).run([train.name], metadata=False)
    ledger.write_bytes(old_state)
    with pytest.raises(ValueError, match="behind or differs"):
        runner(tmp_path, store, sources).run([evaluation.name], metadata=False)
    assert sum(key.endswith(".tar") for key in store.objects) == 1


def test_crash_between_initial_config_and_database_is_recoverable(tmp_path, monkeypatch):
    from src import stage_audioset

    source = source_archive(tmp_path)
    store = FakeS3()
    original = stage_audioset.sqlite3.connect

    def crash(*args, **kwargs):
        raise RuntimeError("crash during first database creation")

    monkeypatch.setattr(stage_audioset.sqlite3, "connect", crash)
    with pytest.raises(RuntimeError, match="first database"):
        runner(tmp_path, store, [source]).run(metadata=False)
    assert (tmp_path / "stage" / "staging-config.json").exists()
    assert not (tmp_path / "stage" / "staging.sqlite3").exists()
    monkeypatch.setattr(stage_audioset.sqlite3, "connect", original)
    assert runner(tmp_path, store, [source]).run(metadata=False)["verified_sources"] == 1


def test_empty_work_directory_from_initialization_crash_can_resume(tmp_path):
    source = source_archive(tmp_path)
    folder = tmp_path / "stage" / "work" / source.name
    folder.mkdir(parents=True)
    assert runner(tmp_path, FakeS3(), [source]).run(metadata=False)["verified_sources"] == 1


def test_deadline_before_index_publication_never_claims_completion(tmp_path, monkeypatch):
    source = source_archive(tmp_path)
    store = FakeS3()
    staged = runner(tmp_path, store, [source])
    publish = staged.publish
    original_index = staged.index

    def deadline_before_publish(reason=None):
        staged.deadline = 0
        return publish(reason)

    def all_payloads_ready(reason=None):
        return {**original_index(reason), "status": "complete"}

    monkeypatch.setattr(staged, "publish", deadline_before_publish)
    monkeypatch.setattr(staged, "index", all_payloads_ready)
    result = staged.run(metadata=False)
    assert result["status"] == "partial"
    assert result["publication_pending"] is True
    assert result["verified_sources"] == 1
    assert result["index_key"] not in store.objects
    assert (tmp_path / "stage" / "work" / source.name).exists()


def test_repack_discards_old_owned_tar_before_writing_new_copy(tmp_path, monkeypatch):
    source = source_archive(tmp_path)
    store = FakeS3()
    store.fail_tar_upload = True
    with pytest.raises(OSError):
        runner(tmp_path, store, [source]).run(metadata=False)
    staged = runner(tmp_path, store, [source])
    original = tarfile.open

    def open_archive(name=None, *args, **kwargs):
        if name and str(name).endswith("prepared.tar.partial"):
            assert not Path(str(name).removesuffix(".partial")).exists()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(tarfile, "open", open_archive)
    store.fail_tar_upload = False
    assert staged.run(metadata=False)["verified_sources"] == 1


def test_changed_prepared_contract_is_rejected_before_reusing_audio(tmp_path):
    source = source_archive(tmp_path)
    prepared = tmp_path / "old-prepared"
    prepared.mkdir()
    (prepared / "preparation-config.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "sample_rate": 16000,
                "minimum_samples": 1,
                "maximum_seconds": 30,
            }
        )
    )
    staged = runner(tmp_path, FakeS3(), [source], reuse_prepared=prepared)
    staged.prepare_callback = None
    with pytest.raises(ValueError, match="different audio preparation contract"):
        staged.run(metadata=False)

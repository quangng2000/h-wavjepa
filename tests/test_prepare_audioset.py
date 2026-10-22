"""Bounded, CPU-only source preparation; fixture archives never use the network."""

import io
import json
import shutil
import subprocess
import tarfile
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from src.prepare_audioset import (
    MIN_SAMPLES,
    Source,
    builtin_sources,
    connect_ledger,
    convert_audio,
    export_manifests,
    inspect_wav,
    member_identity,
    prepare_metadata,
    prepare_source,
    sha256_file,
)


def wav_bytes(samples=MIN_SAMPLES, rate=16_000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x01\x00" * samples * channels)
    return buffer.getvalue()


def archive(
    tmp_path, *, name="unbal_train000", split="train", directory="unbal_train", members=None
):
    path = tmp_path / f"{name}.tar"
    if members is None:
        members = [("aaaaaaaaaaa", wav_bytes()), ("bbbbbbbbbbb", wav_bytes())]
    with tarfile.open(path, "w") as handle:
        for video_id, payload in members:
            item = tarfile.TarInfo(f"audio/{directory}/{video_id}.wav")
            item.size = len(payload)
            handle.addfile(item, io.BytesIO(payload))
    return Source(name, str(path), sha256_file(path), path.stat().st_size, split, directory)


def prepare(source, output, db, **kwargs):
    return prepare_source(source, output, db, minimum_free_bytes=0, **kwargs)


def test_pinned_catalog_retains_all_unbalanced_training_shards():
    sources = builtin_sources()
    assert len(sources) == 889
    assert sum(source.split == "train" for source in sources) == 870
    assert sum(source.split == "val" for source in sources) == 10
    assert sum(source.split == "test" for source in sources) == 9
    assert sources[0].split == "test"
    assert sources[-1].name == "unbal_train869"


def test_capped_source_is_fully_verified_and_resume_is_idempotent(tmp_path):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        result = prepare(source, output, db, max_clips=1)
        assert result["status"] == "partial"
        assert result["verified_sha256"] == source.sha256
        report = export_manifests(output, db, verify_files=True)
        assert report["splits"]["train"]["clips"] == 1
        first_path = output / json.loads((output / "train.jsonl").read_text())["path"]
        first_mtime = first_path.stat().st_mtime_ns
        assert prepare(source, output, db)["status"] == "complete"
        assert prepare(source, output, db)["status"] == "already_complete"
        assert first_path.stat().st_mtime_ns == first_mtime
        report = export_manifests(output, db, verify_files=True)
        assert report["splits"]["train"]["clips"] == 2
        assert len((output / "clips.provenance.jsonl").read_text().splitlines()) == 2
        assert not list((output / ".temporary").iterdir())


def test_full_archive_checksum_failure_excludes_even_valid_decoded_audio(tmp_path):
    source = replace(archive(tmp_path), sha256="0" * 64)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            prepare(source, output, db, max_clips=1)
        report = export_manifests(output, db)
        assert report["splits"]["train"]["clips"] == 0
        assert report["sources"][0]["status"] == "invalid"
        with pytest.raises(ValueError, match="failed integrity"):
            prepare(source, output, db)


def test_interrupted_stream_resumes_without_exposing_unverified_outputs(tmp_path):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        # Simulate committed clip work followed by a process interrupted before
        # archive verification, then replay the same pinned source.
        prepare(source, output, db, max_clips=1)
        db.execute("UPDATE sources SET status='interrupted', verified_sha256=NULL")
        db.commit()
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 0
        prepare(source, output, db)
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 2


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../../outside.wav", tarfile.REGTYPE),
        ("/audio/unbal_train/aaaaaaaaaaa.wav", tarfile.REGTYPE),
        ("audio/unbal_train/aaaaaaaaaaa.wav", tarfile.SYMTYPE),
        ("audio/unbal_train/aaaaaaaaaaa.wav", tarfile.LNKTYPE),
        ("audio/eval/aaaaaaaaaaa.wav", tarfile.REGTYPE),
        ("audio/unbal_train/not-video-id.wav", tarfile.REGTYPE),
        ("audio/unbal_train/aaaaaaaaaaa.json", tarfile.REGTYPE),
    ],
)
def test_unsafe_member_names_types_and_unexpected_layout_are_rejected(tmp_path, name, kind):
    source = archive(tmp_path)
    member = tarfile.TarInfo(name)
    member.size = 123
    member.type = kind
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        member_identity(member, source)
    assert not (tmp_path / "outside.wav").exists()


def test_oversized_member_is_rejected_before_reading_payload(tmp_path):
    source = archive(tmp_path)
    member = tarfile.TarInfo("audio/unbal_train/aaaaaaaaaaa.flac")
    member.size = 33 * 1024**2
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        member_identity(member, source)


def test_split_collision_never_enters_two_manifests(tmp_path):
    test = archive(tmp_path, name="eval00", split="test", directory="eval")
    train = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        prepare(test, output, db)
        with pytest.raises(ValueError, match="Cross-split"):
            prepare(train, output, db)
        report = export_manifests(output, db)
        assert report["splits"]["test"]["clips"] == 2
        assert report["splits"]["train"]["clips"] == 0
        assert report["rejections"][train.name] == 1
        with pytest.raises(ValueError, match="unresolved cross-split"):
            prepare(train, output, db)
        assert "Duplicate recording ID" in (output / "rejections.jsonl").read_text()


def test_changed_source_descriptor_cannot_reuse_existing_ledger(tmp_path):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        prepare(source, output, db)
        with pytest.raises(ValueError, match="changed"):
            prepare(replace(source, split="val"), output, db)


def test_free_space_floor_stops_before_conversion(tmp_path):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        with pytest.raises(OSError, match="Free disk space"):
            prepare_source(source, output, db, minimum_free_bytes=10**20)
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 0


def test_short_and_truncated_outputs_cannot_be_accepted(tmp_path):
    short = tmp_path / "short.wav"
    short.write_bytes(wav_bytes(MIN_SAMPLES - 1))
    with pytest.raises(ValueError, match="no padding allowed"):
        inspect_wav(short, MIN_SAMPLES, 480_000)
    broken = tmp_path / "broken.wav"
    broken.write_bytes(wav_bytes()[:-2])
    with pytest.raises(ValueError, match="Truncated"):
        inspect_wav(broken, MIN_SAMPLES, 480_000)


def test_decoder_uses_bounded_local_ffmpeg_without_padding(monkeypatch, tmp_path):
    original = tmp_path / "source.flac"
    original.write_bytes(b"fake input; subprocess is inspected")
    output = tmp_path / "out.wav"

    def run(command, **kwargs):
        assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"
        assert command[command.index("-t") + 1] == "31"
        assert "apad" not in " ".join(command)
        assert kwargs["timeout"] == 120
        assert command[command.index("-f") + 1] == "flac"
        Path(command[-1]).write_bytes(wav_bytes())

    monkeypatch.setattr(subprocess, "run", run)
    result = convert_audio(
        original, output, ffmpeg="ffmpeg", minimum_samples=MIN_SAMPLES, maximum_seconds=30
    )
    assert result["samples"] == MIN_SAMPLES
    assert result["sha256"] == sha256_file(output)
    assert not original.exists()
    assert not output.with_suffix(".wav.partial").exists()


def test_failed_decoder_cleans_temporary_files(monkeypatch, tmp_path):
    original = tmp_path / "source.flac"
    original.write_bytes(b"bad flac")
    output = tmp_path / "out.wav"

    def run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        convert_audio(
            original, output, ffmpeg="ffmpeg", minimum_samples=MIN_SAMPLES, maximum_seconds=30
        )
    assert not original.exists()
    assert not output.exists()
    assert not output.with_suffix(".wav.partial").exists()


def test_corrupted_prepared_file_is_detected_by_verify_only(tmp_path):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        prepare(source, output, db)
        export_manifests(output, db)
        path = output / json.loads((output / "train.jsonl").read_text().splitlines()[0])["path"]
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
        with pytest.raises(ValueError, match="checksum mismatch"):
            export_manifests(output, db, verify_files=True)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg integration runs on preparation CPU host"
)
def test_real_ffmpeg_resamples_stereo_and_preserves_duration(tmp_path):
    source = tmp_path / "stereo.wav"
    source.write_bytes(wav_bytes(480_000, rate=48_000, channels=2))
    output = tmp_path / "mono.wav"
    result = convert_audio(
        source, output, ffmpeg="ffmpeg", minimum_samples=MIN_SAMPLES, maximum_seconds=30
    )
    assert result["samples"] == 160_000


def test_failed_resume_retains_previously_verified_outputs(tmp_path, monkeypatch):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        prepare(source, output, db, max_clips=1)
        from src import prepare_audioset

        def fail(source):
            raise OSError("network unavailable")

        monkeypatch.setattr(prepare_audioset, "open_source", fail)
        with pytest.raises(OSError, match="network unavailable"):
            prepare(source, output, db)
        report = export_manifests(output, db)
        assert report["splits"]["train"]["clips"] == 1
        assert report["sources"][0]["status"] == "interrupted"


def test_metadata_is_verified_and_repeat_download_is_skipped(tmp_path, monkeypatch):
    from src.prepare_audioset import REVISION

    files = []
    for name in (
        "unbalanced_train_segments.csv",
        "balanced_train_segments.csv",
        "eval_segments.csv",
    ):
        source = tmp_path / name
        source.write_text(
            '# YTID, start_seconds, end_seconds, positive_labels\nabcdefghijk, 30, 40, "/m/example"\n'
        )
        files.append(
            {
                "path": f"data/{name}",
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
                "url": str(source),
            }
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"revision": REVISION, "files": files}))
    output = tmp_path / "prepared"
    output.mkdir()
    report = prepare_metadata(catalog, output)
    assert len(report["files"]) == 3
    for entry in files:
        Path(entry["url"]).unlink()
    assert prepare_metadata(catalog, output) == report
    assert "30, 40" in (output / "metadata" / "eval_segments.csv").read_text()


def test_member_read_failure_removes_unscheduled_temporary_input(tmp_path, monkeypatch):
    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()

    class BrokenPayload(io.BytesIO):
        def read(self, size=-1):
            raise OSError("broken member stream")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda self, member: BrokenPayload())
    with connect_ledger(output) as db:
        with pytest.raises(OSError, match="broken member stream"):
            prepare(source, output, db)
        assert not list((output / ".temporary").iterdir())
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 0


def test_interrupted_resume_never_publishes_new_unverified_clips(tmp_path, monkeypatch):
    from src.prepare_audioset import HashingReader

    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    with connect_ledger(output) as db:
        prepare(source, output, db, max_clips=1)

        def fail(self, expected_sha256):
            raise OSError("network failed near archive end")

        monkeypatch.setattr(HashingReader, "finish", fail)
        with pytest.raises(OSError, match="network failed"):
            prepare(source, output, db)
        assert db.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 2
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 1


@pytest.mark.parametrize(
    "failure", [OSError(28, "No space left on device"), subprocess.TimeoutExpired("ffmpeg", 120)]
)
def test_infrastructure_failure_is_retryable_instead_of_rejected(tmp_path, monkeypatch, failure):
    from src import prepare_audioset

    source = archive(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    original = prepare_audioset.convert_audio

    def fail(input_path, *args, **kwargs):
        input_path.unlink()
        raise failure

    with connect_ledger(output) as db:
        monkeypatch.setattr(prepare_audioset, "convert_audio", fail)
        with pytest.raises(type(failure)):
            prepare(source, output, db, workers=1)
        assert db.execute("SELECT COUNT(*) FROM members").fetchone()[0] == 0
        report = export_manifests(output, db)
        assert report["sources"][0]["status"] == "interrupted"
        assert report["rejections"] == {}
        monkeypatch.setattr(prepare_audioset, "convert_audio", original)
        assert prepare(source, output, db)["status"] == "complete"
        assert export_manifests(output, db)["splits"]["train"]["clips"] == 2


def test_ffmpeg_disk_error_is_an_infrastructure_failure(tmp_path, monkeypatch):
    source = tmp_path / "input.flac"
    source.write_bytes(b"fake flac")
    output = tmp_path / "output.wav"

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr=b"av_interleaved_write_frame(): No space left on device"
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(OSError, match="infrastructure failure"):
        convert_audio(
            source, output, ffmpeg="ffmpeg", minimum_samples=MIN_SAMPLES, maximum_seconds=30
        )
    assert not source.exists()
    assert not output.exists()


def test_invalid_decoder_input_is_a_recorded_rejection(tmp_path, monkeypatch):
    source = archive(tmp_path, members=[("aaaaaaaaaaa", b"not valid audio")])
    output = tmp_path / "prepared"
    output.mkdir()

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(
            1, command, stderr=b"Invalid data found when processing input"
        )

    monkeypatch.setattr(subprocess, "run", run)
    with connect_ledger(output) as db:
        assert prepare(source, output, db)["status"] == "complete"
        report = export_manifests(output, db)
        assert report["rejections"][source.name] == 1
        assert report["splits"]["train"]["clips"] == 0

"""Verified local-image input avoids network waits without weakening archive checks."""

import hashlib
import io
import tarfile

import pytest

from scripts import benchmark_gpu as benchmark


def archive(tmp_path, monkeypatch, *, unsafe=None):
    path = tmp_path / "image-pilot.tar"
    with tarfile.open(path, "w") as output:
        for name, raw in [
            ("train.jsonl", b'{"path":"audio/example.wav"}\n'),
            ("val.jsonl", b""),
            ("test.jsonl", b""),
            ("audio/example.wav", b"fixture payload"),
            *([] if unsafe is None else [(unsafe, b"must not escape")]),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            output.addfile(info, io.BytesIO(raw))
    original = path.read_bytes()
    monkeypatch.setattr(benchmark, "PILOT_BYTES", len(original))
    monkeypatch.setattr(benchmark, "PILOT_SHA", hashlib.sha256(original).hexdigest())
    monkeypatch.setenv("HWAVJEPA_LOCAL_PILOT", str(path))
    return path, original


def test_local_pilot_verifies_and_extracts_without_network_or_deleting_source(
    tmp_path, monkeypatch
):
    source, original = archive(tmp_path, monkeypatch)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("A local pilot must not create an S3 client")

    monkeypatch.setattr("src.stage_audioset.make_client", network_forbidden)
    destination = tmp_path / "owned-output"
    benchmark.prepare_pilot(destination)
    assert (destination / "audio/example.wav").read_bytes() == b"fixture payload"
    assert source.read_bytes() == original
    assert not (destination / "pilot.tar").exists()


@pytest.mark.parametrize("failure", ["size", "sha"])
def test_bad_local_pilot_rejected_before_extraction_and_original_preserved(
    tmp_path, monkeypatch, failure
):
    source, original = archive(tmp_path, monkeypatch)
    if failure == "size":
        monkeypatch.setattr(benchmark, "PILOT_BYTES", len(original) + 1)
    else:
        monkeypatch.setattr(benchmark, "PILOT_SHA", "0" * 64)
    destination = tmp_path / "owned-output"
    with pytest.raises(ValueError, match="size or SHA256"):
        benchmark.prepare_pilot(destination)
    assert not destination.exists()
    assert source.read_bytes() == original


def test_local_pilot_cannot_escape_owned_directory_or_erase_existing_data(tmp_path, monkeypatch):
    source, original = archive(tmp_path, monkeypatch, unsafe="../escape.txt")
    destination = tmp_path / "owned-output"
    with pytest.raises(ValueError, match="unsafe path"):
        benchmark.prepare_pilot(destination)
    assert not destination.exists() and not (tmp_path / "escape.txt").exists()
    assert source.read_bytes() == original
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError):
        benchmark.prepare_pilot(destination)
    assert sentinel.read_text() == "existing data"


def test_s3_fallback_still_verifies_extracts_and_closes_body(tmp_path, monkeypatch):
    _, original = archive(tmp_path, monkeypatch)
    monkeypatch.delenv("HWAVJEPA_LOCAL_PILOT")
    for key, value in {
        "HWAVJEPA_S3_ENDPOINT": "https://storage.example.invalid",
        "HWAVJEPA_PILOT_BUCKET": "private",
        "HWAVJEPA_PILOT_KEY": "pilot.tar",
    }.items():
        monkeypatch.setenv(key, value)

    class Body(io.BytesIO):
        def iter_chunks(self, size):
            yield from iter(lambda: self.read(size), b"")

    body = Body(original)

    class Client:
        def get_object(self, **kwargs):
            assert kwargs == {"Bucket": "private", "Key": "pilot.tar"}
            return {"Body": body}

    monkeypatch.setattr("src.stage_audioset.make_client", lambda _: Client())
    destination = tmp_path / "download-output"
    benchmark.prepare_pilot(destination)
    assert body.closed
    assert (destination / "audio/example.wav").read_bytes() == b"fixture payload"
    assert not (destination / "pilot.tar").exists()

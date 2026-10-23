"""Offline Nat conversion tests use small analytic archive fixtures only."""

import hashlib
import io
import json
import struct
import sys
import tarfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.prepare_nat_assets import (
    MAX_BRIR_MEMBER_BYTES,
    PreparationLimit,
    _npy_array,
    assemble_nat_manifest,
    convert_scene_array,
    main,
    prepare_brir_shard,
    prepare_wham_noise,
)


def _scene(noise_indices=(1,), length=320):
    array = np.zeros((6, 2, length), dtype=np.float64)
    for index in (0, *noise_indices):
        array[index, 0, 8] = 1 + index
        array[index, 1, 12] = (1 + index) / 2
        array[index, :, -12] = 0.3
    return array


def _npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _tar(tmp_path, entries, filename="source.tar"):
    path = tmp_path / filename
    with tarfile.open(path, "w") as archive:
        for name, value in entries:
            info = tarfile.TarInfo(name)
            payload = _npy(value) if isinstance(value, np.ndarray) else value
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _convert(tmp_path, entries, **kwargs):
    source, digest = _tar(tmp_path, entries)
    directory = tmp_path / "converted"
    result = prepare_brir_shard(
        source,
        digest,
        directory,
        source_member="train/webdataset_output/shard-000000.tar",
        **kwargs,
    )
    return directory, result


def test_brir_manifest_keeps_matched_scene_indices_and_verified_objects(tmp_path):
    root, report = _convert(
        tmp_path,
        [("localized.npy", _scene((2,))), ("diffuse.npy", _scene((1, 3, 5)))],
    )
    assert report["status"] == "complete"
    assert report["converted_count"] == 2
    assert report["split"] == "train"
    assert report["source"]["verified"]
    assert report["scenes"][0]["source_indices"] == [0, 2]
    assert report["scenes"][1]["source_indices"] == [0, 1, 3, 5]
    assert [record["noise_type"] for record in report["scenes"]] == ["localized", "diffuse"]
    for scene in report["scenes"]:
        logicals = [scene["source_rir"], *scene["noise_rirs"]]
        records = [report["assets"][name] for name in logicals]
        assert len({record["key"] for record in records}) == 1
        assert [record["index"] for record in records] == list(range(len(logicals)))
        object_path = root / records[0]["key"]
        assert hashlib.sha256(object_path.read_bytes()).hexdigest() == records[0]["sha256"]
        assert object_path.stat().st_size == records[0]["bytes"]
        with np.load(object_path, allow_pickle=False) as archive:
            array = archive["arr_0"]
        assert array.shape == (len(logicals), 2, 160)
        assert array.dtype == np.float32
        np.testing.assert_allclose(array[:, 1, 6], array[:, 0, 4] / 2, rtol=0.002)
    assert json.loads((root / "manifest.json").read_text()) == report


def test_resampling_preserves_three_second_tail_and_rejects_aliasing():
    full = _scene(length=96000)
    converted, _, _ = convert_scene_array(full)
    assert converted.shape == (2, 2, 48000)
    assert np.abs(converted[..., -15:]).max() > 0.1
    # A 12 kHz component is above the target Nyquist frequency and must be
    # strongly suppressed; naive stride-2 decimation would retain its energy.
    time = np.arange(32000, dtype=np.float64) / 32000
    source = _scene(length=32000)
    source[0, 0] = np.sin(2 * np.pi * 1000 * time)
    source[0, 1] = np.sin(2 * np.pi * 12000 * time)
    result, _, _ = convert_scene_array(source)
    assert np.std(result[0, 0, 100:-100]) > 0.6
    assert np.std(result[0, 1, 100:-100]) < 0.005


@pytest.mark.parametrize("indices", [(), (1, 2)])
def test_invalid_noise_population_is_rejected(indices):
    with pytest.raises(ValueError, match="one localized or three to five"):
        convert_scene_array(_scene(indices))


@pytest.mark.parametrize("indices", [(1,), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)])
def test_noise_population_classification(indices):
    result, kind, source_indices = convert_scene_array(_scene(indices))
    assert len(result) == 1 + len(indices)
    assert kind == ("localized" if len(indices) == 1 else "diffuse")
    assert source_indices == [0, *indices]


@pytest.mark.parametrize("bad", ["nan", "zero_source", "float32", "shape", "too_long"])
def test_invalid_scene_data_fails_closed(bad):
    array = _scene()
    if bad == "nan":
        array[0, 0, 0] = np.nan
    elif bad == "zero_source":
        array[0] = 0
    elif bad == "float32":
        array = array.astype(np.float32)
    elif bad == "shape":
        array = array[:2]
    else:
        array = np.zeros((6, 2, 96001), dtype=np.float64)
    with pytest.raises(ValueError):
        convert_scene_array(array)


def test_bad_source_hash_creates_no_output(tmp_path):
    source, _ = _tar(tmp_path, [("one.npy", _scene())])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_brir_shard(
            source,
            "0" * 64,
            output,
            source_member="train/webdataset_output/shard-000000.tar",
        )
    assert not output.exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    source, digest = _tar(tmp_path, [("one.npy", _scene())])
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "precious"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare_brir_shard(
            source,
            digest,
            output,
            source_member="train/webdataset_output/shard-000000.tar",
        )
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("name", ["../outside.npy", "/absolute.npy", "nested/a.npy", "a\\b.npy"])
def test_tar_member_paths_cannot_escape_output(tmp_path, name):
    with pytest.raises(ValueError):
        _convert(tmp_path, [(name, _scene())])
    assert list((tmp_path / "converted" / "objects").iterdir()) == []


def test_duplicate_scene_does_not_overwrite_verified_first_scene(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        _convert(tmp_path, [("same.npy", _scene()), ("same.npy", _scene((1, 2, 3)))])
    root = tmp_path / "converted"
    report = json.loads((root / "manifest.json").read_text())
    assert report["status"] == "failed" and report["converted_count"] == 1
    record = report["assets"][report["scenes"][0]["source_rir"]]
    assert hashlib.sha256((root / record["key"]).read_bytes()).hexdigest() == record["sha256"]


@pytest.mark.parametrize("bound", ["max_scenes", "max_archive_members"])
def test_count_bound_preserves_partial_manifest_and_objects(tmp_path, bound):
    with pytest.raises(PreparationLimit):
        _convert(tmp_path, [("one.npy", _scene()), ("two.npy", _scene())], **{bound: 1})
    report = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    assert report["status"] == "partial"
    assert report["converted_count"] == 1
    assert len(list((tmp_path / "converted" / "objects").iterdir())) == 1


def test_output_byte_bound_leaves_partial_manifest_without_incomplete_objects(tmp_path):
    with pytest.raises(PreparationLimit, match="byte bound"):
        _convert(tmp_path, [("one.npy", _scene())], max_output_bytes=1)
    root = tmp_path / "converted"
    report = json.loads((root / "manifest.json").read_text())
    assert report["status"] == "partial"
    assert report["converted_count"] == report["output_bytes"] == 0
    assert not list((root / "objects").iterdir())
    assert not list(root.glob(".scene-*"))


def test_tar_symlink_is_rejected_without_following_it(tmp_path):
    path = tmp_path / "source.tar"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("one.npy")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(ValueError, match="regular NPY"):
        prepare_brir_shard(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            tmp_path / "output",
            source_member="train/webdataset_output/shard-000000.tar",
        )


def test_numpy_huge_shape_is_rejected_before_allocation():
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": "<f8", "fortran_order": False, "shape": (6, 2, 10**12)},
    )
    with pytest.raises(ValueError, match="96000"):
        _npy_array(stream.getvalue())


def test_numpy_truncated_and_pickle_inputs_are_rejected():
    with pytest.raises(ValueError, match="payload length"):
        _npy_array(_npy(_scene())[:-8])
    stream = io.BytesIO()
    np.save(stream, np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="float64"):
        _npy_array(stream.getvalue())


def test_tar_member_size_checked_before_content_read(tmp_path, monkeypatch):
    source, digest = _tar(tmp_path, [("one.npy", _scene())])
    original = tarfile.TarFile.next

    def oversized(self):
        member = original(self)
        if member is not None:
            member.size = MAX_BRIR_MEMBER_BYTES + 1
        return member

    monkeypatch.setattr(tarfile.TarFile, "next", oversized)
    with pytest.raises(ValueError, match="size.*exceeds"):
        prepare_brir_shard(
            source,
            digest,
            tmp_path / "output",
            source_member="train/webdataset_output/shard-000000.tar",
        )


def _float_wav(samples, *, rate=16000, subtype="FLOAT"):
    stream = io.BytesIO()
    sf.write(stream, np.asarray(samples), rate, subtype=subtype, format="WAV")
    return stream.getvalue()


def _wham_zip(tmp_path, entries):
    path = tmp_path / "noise.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, samples in entries:
            archive.writestr(name, samples if isinstance(samples, bytes) else _float_wav(samples))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_wham_selects_train_only_and_preserves_first_channel_duration(tmp_path):
    first = np.linspace(-0.8, 0.8, 161, dtype=np.float32)
    stereo = np.stack([first, -first], axis=-1)
    source, digest = _wham_zip(
        tmp_path,
        [
            ("wham_noise/tr/train.wav", stereo),
            ("wham_noise/cv/heldout.wav", stereo),
            ("wham_noise/tt/test.wav", stereo),
        ],
    )
    root = tmp_path / "converted"
    report = prepare_wham_noise(source, digest, root)
    assert report["status"] == "complete"
    assert report["converted_count"] == 1
    record = report["assets"]["noise/train.wav"]
    assert record["peak_gain"] == 1.0
    with wave.open(str(root / record["key"]), "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        assert handle.getframerate() == 16000 and handle.getnframes() == 161
        decoded = np.frombuffer(handle.readframes(161), dtype="<i2") / 32768.0
    np.testing.assert_allclose(decoded, first, atol=1 / 32768)
    assert hashlib.sha256((root / record["key"]).read_bytes()).hexdigest() == record["sha256"]


def test_wham_peak_gain_is_recorded_without_clipping(tmp_path):
    source, digest = _wham_zip(
        tmp_path,
        [
            ("wham_noise/tr/loud.wav", np.array([[2.0, 0.0], [-1.0, 0.0]], dtype=np.float32)),
        ],
    )
    root = tmp_path / "converted"
    report = prepare_wham_noise(source, digest, root)
    record = report["assets"]["noise/loud.wav"]
    assert record["peak_gain"] == pytest.approx((32767 / 32768) / 2)
    with wave.open(str(root / record["key"]), "rb") as handle:
        decoded = np.frombuffer(handle.readframes(2), dtype="<i2")
    assert decoded[0] == 32767 and decoded[1] == -16384


@pytest.mark.parametrize("bad", ["rate", "mono", "pcm", "nan", "silence", "long"])
def test_wham_invalid_audio_is_rejected(tmp_path, bad):
    stereo = np.ones((32, 2), dtype=np.float32) * 0.1
    if bad == "mono":
        stereo = stereo[:, 0]
    elif bad == "nan":
        stereo[0, 0] = np.nan
    elif bad == "silence":
        stereo[:] = 0
    elif bad == "long":
        stereo = np.ones((60 * 16000 + 1, 2), dtype=np.float32) * 0.1
    payload = _float_wav(
        stereo, rate=8000 if bad == "rate" else 16000, subtype="PCM_16" if bad == "pcm" else "FLOAT"
    )
    source, digest = _wham_zip(tmp_path, [("wham_noise/tr/bad.wav", payload)])
    with pytest.raises(ValueError):
        prepare_wham_noise(source, digest, tmp_path / "converted")


def test_wham_count_limit_keeps_first_completed_noise(tmp_path):
    stereo = np.stack([np.linspace(-0.1, 0.1, 32, dtype=np.float32)] * 2, axis=-1)
    source, digest = _wham_zip(
        tmp_path,
        [
            ("wham_noise/tr/a.wav", stereo),
            ("wham_noise/tr/b.wav", stereo),
        ],
    )
    with pytest.raises(PreparationLimit, match="file count"):
        prepare_wham_noise(source, digest, tmp_path / "converted", max_files=1)
    report = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    assert report["status"] == "partial" and report["converted_count"] == 1


def test_wham_traversal_is_rejected_without_extracting(tmp_path):
    source, digest = _wham_zip(tmp_path, [("wham_noise/tr/../../outside.wav", b"bad")])
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_wham_noise(source, digest, tmp_path / "converted")
    assert not (tmp_path / "outside.wav").exists()


def test_zip_directory_size_is_bounded_before_zipfile_allocates(tmp_path, monkeypatch):
    source = tmp_path / "oversized.zip"
    source.write_bytes(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, 128 * 1024**2, 0, 0))

    def must_not_open(*args, **kwargs):
        raise AssertionError("ZipFile must not parse oversized metadata")

    monkeypatch.setattr(zipfile, "ZipFile", must_not_open)
    with pytest.raises(PreparationLimit, match="central directory"):
        prepare_wham_noise(
            source, hashlib.sha256(source.read_bytes()).hexdigest(), tmp_path / "converted"
        )


def test_only_official_training_shard_numbers_are_accepted(tmp_path):
    source, digest = _tar(tmp_path, [("one.npy", _scene())])
    with pytest.raises(ValueError, match="index"):
        prepare_brir_shard(
            source,
            digest,
            tmp_path / "converted",
            source_member="train/webdataset_output/shard-000070.tar",
        )
    assert not (tmp_path / "converted").exists()


@pytest.mark.parametrize("case", ["below_pcm_lsb", "constant_first_ten_seconds"])
def test_wham_rejects_noise_that_becomes_constant_in_renderer(tmp_path, case):
    if case == "below_pcm_lsb":
        first = np.linspace(-1e-7, 1e-7, 128, dtype=np.float32)
    else:
        first = np.zeros(160001, dtype=np.float32)
        first[-1] = 0.5
    source, digest = _wham_zip(
        tmp_path, [("wham_noise/tr/invalid.wav", np.stack([first, first], axis=-1))]
    )
    with pytest.raises(ValueError, match="centered variance"):
        prepare_wham_noise(source, digest, tmp_path / "converted")


def _assembly_inputs(tmp_path):
    scene_dir = tmp_path / "scene-input"
    scene_dir.mkdir()
    brir_root, _ = _convert(
        scene_dir, [("local.npy", _scene()), ("diffuse.npy", _scene((1, 2, 3)))]
    )
    mono = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    source, digest = _wham_zip(
        tmp_path, [("wham_noise/tr/first.wav", np.stack([mono, -mono], axis=-1))]
    )
    noise_root = tmp_path / "noise-output"
    prepare_wham_noise(source, digest, noise_root)
    return brir_root / "manifest.json", noise_root / "manifest.json"


def _assemble(tmp_path, brir, noise, **kwargs):
    options = {
        "endpoint": "https://storage.example.com",
        "bucket": "nat-assets",
        "prefix": "nat/verified-v1",
        "allow_pilot": True,
        **kwargs,
    }
    return assemble_nat_manifest([brir], noise, tmp_path / "remote.json", **options)


def test_assembled_remote_manifest_loads_without_network_and_has_upload_layout(
    tmp_path, monkeypatch
):
    from src.data.naturalistic_audio import NaturalisticAudio

    brir, noise = _assembly_inputs(tmp_path)
    report = _assemble(tmp_path, brir, noise)
    manifest = json.loads((tmp_path / "remote.json").read_text())
    assert report["status"] == "offline_verified_not_uploaded"
    assert report["coverage"] == "pilot"
    assert report["scene_count"] == 2 and report["noise_count"] == 1
    assert manifest["provenance"]["remote_verified"] is False

    def no_network(*args, **kwargs):
        raise AssertionError("Assembly and constructor must not fetch remote assets")

    monkeypatch.setattr(NaturalisticAudio, "_materialize", no_network)
    renderer = NaturalisticAudio(
        tmp_path / "remote.json", cache_dir=tmp_path / "cache", require_asset_checksums=True
    )
    assert len(renderer.scenes["localized"]) == len(renderer.scenes["diffuse"]) == 1
    plan = json.loads((tmp_path / "remote.json.upload.json").read_text())
    assert len(plan["objects"]) == 3
    assert plan["manifest"]["key"] == "nat/verified-v1/manifest.json"
    assert (
        plan["manifest"]["sha256"]
        == hashlib.sha256((tmp_path / "remote.json").read_bytes()).hexdigest()
    )
    for record in plan["objects"]:
        assert record["key"].startswith("nat/verified-v1/objects/")
        payload = Path(record["local_path"]).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "test_split",
        "wrong_revision",
        "source_not_verified",
        "duplicate_scene",
        "missing_asset",
        "unexpected_asset",
        "wrong_index",
        "traversal",
        "wrong_count",
        "wrong_bytes",
    ],
)
def test_assembly_rejects_inconsistent_components_before_publication(tmp_path, mutation):
    brir, noise = _assembly_inputs(tmp_path)
    content = json.loads(brir.read_text())
    first = next(iter(content["assets"]))
    if mutation == "partial":
        content["status"] = "partial"
    elif mutation == "test_split":
        content["split"] = "test"
    elif mutation == "wrong_revision":
        content["provenance"]["revision"] = "0" * 40
    elif mutation == "source_not_verified":
        content["source"]["verified"] = False
    elif mutation == "duplicate_scene":
        content["scenes"][1]["id"] = content["scenes"][0]["id"]
    elif mutation == "missing_asset":
        del content["assets"][first]
    elif mutation == "unexpected_asset":
        content["assets"]["extra.npz"] = content["assets"][first]
    elif mutation == "wrong_index":
        content["assets"][first]["index"] = 5
    elif mutation == "traversal":
        content["assets"][first]["key"] = "../outside.npz"
    elif mutation == "wrong_count":
        content["converted_count"] += 1
    else:
        content["output_bytes"] += 1
    brir.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        _assemble(tmp_path, brir, noise)
    assert not (tmp_path / "remote.json").exists()
    assert not (tmp_path / "remote.json.upload.json").exists()


def test_assembly_checks_local_object_checksum_and_preserves_inputs(tmp_path):
    brir, noise = _assembly_inputs(tmp_path)
    content = json.loads(brir.read_text())
    record = next(iter(content["assets"].values()))
    physical = brir.parent / record["key"]
    payload = bytearray(physical.read_bytes())
    payload[-1] ^= 1
    physical.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        _assemble(tmp_path, brir, noise)
    assert physical.read_bytes() == bytes(payload)


def test_assembly_rejects_duplicate_component_paths(tmp_path):
    brir, noise = _assembly_inputs(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        assemble_nat_manifest(
            [brir, brir],
            noise,
            tmp_path / "remote.json",
            endpoint="https://storage.example.com",
            bucket="nat-assets",
            prefix="nat/v1",
            allow_pilot=True,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"endpoint": "http://example.com"},
        {"endpoint": "https://user:password@example.com"},
        {"prefix": "../outside"},
        {"prefix": "nat//v1"},
        {"bucket": "bad/bucket"},
    ],
)
def test_assembly_rejects_unsafe_destinations(tmp_path, override):
    brir, noise = _assembly_inputs(tmp_path)
    with pytest.raises(ValueError):
        _assemble(tmp_path, brir, noise, **override)


def test_assembly_requires_explicit_pilot_for_subset(tmp_path):
    brir, noise = _assembly_inputs(tmp_path)
    with pytest.raises(ValueError, match="pinned inventory|1,000|70,000|20,000"):
        _assemble(tmp_path, brir, noise, allow_pilot=False)
    assert not (tmp_path / "remote.json").exists()


def test_assembly_never_overwrites_existing_manifest(tmp_path):
    brir, noise = _assembly_inputs(tmp_path)
    existing = tmp_path / "remote.json"
    existing.write_text("retain this")
    with pytest.raises(FileExistsError):
        _assemble(tmp_path, brir, noise)
    assert existing.read_text() == "retain this"


def test_assembly_object_budget_prevents_publication(tmp_path):
    brir, noise = _assembly_inputs(tmp_path)
    with pytest.raises(PreparationLimit, match="byte bound"):
        _assemble(tmp_path, brir, noise, max_total_object_bytes=1)
    assert not (tmp_path / "remote.json").exists()


def test_assembly_cli_produces_an_immutable_upload_plan(tmp_path, monkeypatch, capsys):
    brir, noise = _assembly_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_nat_assets",
            "assemble",
            "--brir-manifest",
            str(brir),
            "--noise-manifest",
            str(noise),
            "--output-manifest",
            str(tmp_path / "remote.json"),
            "--endpoint",
            "https://storage.example.com",
            "--bucket",
            "nat-assets",
            "--prefix",
            "nat/pilot",
            "--allow-pilot",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["coverage"] == "pilot"
    assert report["status"] == "offline_verified_not_uploaded"
    assert Path(report["upload_plan"]).is_file()

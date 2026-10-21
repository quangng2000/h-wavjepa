import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from src.data.wav_datamodule import (
    WAVManifestDataModule,
    WAVManifestDataset,
    _manifest_paths,
    validate_manifests,
)


def wav(path, frames=16):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(struct.pack(f"<{frames}h", *range(frames)))
    return path


def manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def manifests(tmp_path):
    return {
        f"{split}_manifest": manifest(
            tmp_path / f"{split}.jsonl", [{"path": wav(tmp_path / f"{split}.wav").name}]
        )
        for split in ("train", "val", "test")
    }


def test_hardlink_duplicates_rejected_within_manifest(tmp_path):
    source = wav(tmp_path / "source.wav")
    alias = tmp_path / "hardlink.wav"
    os.link(source, alias)
    path = manifest(tmp_path / "data.jsonl", [{"path": str(source)}, {"path": str(alias)}])
    with pytest.raises(ValueError, match="Duplicate physical"):
        _manifest_paths(path)


def test_hardlink_duplicates_rejected_across_splits(tmp_path, manifests):
    alias = tmp_path / "alias.wav"
    os.link(tmp_path / "train.wav", alias)
    manifests["val_manifest"] = manifest(tmp_path / "val.jsonl", [{"path": str(alias)}])
    with pytest.raises(ValueError, match="Physical WAV files overlap"):
        WAVManifestDataModule(**manifests).setup()


@pytest.mark.parametrize("field", ["group_id", "recording_id"])
def test_declared_group_overlap_is_rejected(tmp_path, manifests, field):
    for split in ("train", "val", "test"):
        manifest(tmp_path / f"{split}.jsonl", [{"path": f"{split}.wav", field: "same-source"}])
    with pytest.raises(ValueError, match=f"{field} overlap"):
        validate_manifests(**manifests)


@pytest.mark.parametrize("field", ["group_id", "recording_id"])
def test_partial_metadata_is_rejected(tmp_path, manifests, field):
    manifest(tmp_path / "train.jsonl", [{"path": "train.wav", field: "speaker-1"}])
    with pytest.raises(ValueError, match=f"{field} is required on every record"):
        WAVManifestDataModule(**manifests).setup()


def test_metadata_disjointness_allows_same_group_within_split(tmp_path, manifests):
    for split in ("train", "val", "test"):
        rows = [{"path": f"{split}.wav", "group_id": split, "recording_id": f"rec-{split}"}]
        if split == "train":
            rows.append(
                {
                    "path": wav(tmp_path / "another.wav").name,
                    "group_id": split,
                    "recording_id": f"rec-{split}",
                }
            )
        manifest(tmp_path / f"{split}.jsonl", rows)
    report = validate_manifests(**manifests, num_samples=8)
    assert report["checks"]["group_fields_disjoint"] == ["group_id", "recording_id"]
    assert report["splits"]["train"] == {
        "files": 2,
        "source_frames": 32,
        "cropped_files": 2,
        "padded_files": 0,
    }
    assert not report["checks"]["content_hashes"]


@pytest.mark.parametrize("identifier", ["", "  ", None, 0, [], {}])
def test_invalid_identifier_has_manifest_line_context(tmp_path, identifier):
    path = manifest(tmp_path / "data.jsonl", [{"path": "a.wav", "group_id": identifier}])
    with pytest.raises(ValueError, match=r"data.jsonl:1: group_id must be a nonempty string"):
        _manifest_paths(path)


def test_missing_audio_has_manifest_line_context(tmp_path):
    path = manifest(tmp_path / "data.jsonl", [{"path": "absent.wav"}])
    with pytest.raises(ValueError, match=r"data.jsonl:1: cannot access WAV file.*absent.wav"):
        _manifest_paths(path)


def test_unreadable_manifest_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="Cannot read UTF-8 JSONL manifest"):
        _manifest_paths(tmp_path / "missing.jsonl")


def test_truncated_payload_rejected_before_loading_even_if_crop_would_fit(tmp_path):
    audio = wav(tmp_path / "truncated.wav", frames=128)
    audio.write_bytes(audio.read_bytes()[:-2])
    with pytest.raises(ValueError, match="Truncated or misaligned PCM sample data"):
        WAVManifestDataset([audio], num_samples=8)


def test_incomplete_final_sample_rejected_before_loading(tmp_path):
    audio = wav(tmp_path / "odd.wav", frames=16)
    raw = bytearray(audio.read_bytes())
    # Standard PCM fixture: change both RIFF and data sizes, append one partial sample.
    struct.pack_into("<I", raw, 4, len(raw) - 8 + 1)
    struct.pack_into("<I", raw, 40, 16 * 2 + 1)
    audio.write_bytes(raw + b"\x00")
    with pytest.raises(ValueError, match="Truncated or misaligned PCM sample data"):
        WAVManifestDataset([audio])


def test_invalid_header_is_actionable(tmp_path):
    audio = tmp_path / "corrupt.wav"
    audio.write_bytes(b"not a wave file")
    with pytest.raises(ValueError, match=r"Cannot read valid PCM WAV.*corrupt.wav"):
        WAVManifestDataset([audio])


def test_failed_setup_does_not_partially_assign_datasets(tmp_path, manifests):
    (tmp_path / "test.wav").write_bytes(b"corrupt")
    module = WAVManifestDataModule(**manifests)
    with pytest.raises(ValueError):
        module.setup()
    assert module.data_train is None
    assert module.data_val is None
    assert module.data_test is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"persistent_workers": True},
        {"prefetch_factor": 2},
        {"prefetch_factor": 0, "num_workers": 1},
        {"num_workers": -1},
        {"num_workers": 0.5},
        {"batch_size": True},
        {"sample_rate": 0},
        {"num_samples": 1.5},
        {"pin_memory": "true"},
        {"persistent_workers": 1},
    ],
)
def test_invalid_loader_configuration_rejected_early(manifests, kwargs):
    with pytest.raises(ValueError):
        WAVManifestDataModule(**manifests, **kwargs)


def test_loader_passes_standard_torch_options(manifests):
    module = WAVManifestDataModule(
        **manifests, num_workers=2, persistent_workers=True, prefetch_factor=3, pin_memory=True
    )
    module.setup()
    loader = module.train_dataloader()
    assert loader.num_workers == 2
    assert loader.persistent_workers
    assert loader.prefetch_factor == 3
    assert loader.pin_memory


def test_single_process_loader_works_with_default_prefetch(manifests):
    module = WAVManifestDataModule(**manifests)
    module.setup()
    loader = module.train_dataloader()
    assert loader.prefetch_factor is None
    assert next(iter(loader))["waveform"].shape == (1, 1024)


def test_validate_cli_is_read_only_and_emits_json(manifests, tmp_path):
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate_data",
            "--train",
            manifests["train_manifest"],
            "--val",
            manifests["val_manifest"],
            "--test",
            manifests["test_manifest"],
            "--num-samples",
            "32",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["valid"]
    assert report["splits"]["test"]["padded_files"] == 1
    assert report["checks"]["group_fields_disjoint"] == []
    assert before == {path: path.read_bytes() for path in tmp_path.iterdir()}


def test_validate_cli_failure_is_nonzero_json(manifests, tmp_path):
    (tmp_path / "train.wav").write_bytes(b"corrupt")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validate_data",
            "--train",
            manifests["train_manifest"],
            "--val",
            manifests["val_manifest"],
            "--test",
            manifests["test_manifest"],
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert not json.loads(result.stderr)["valid"]
    assert "train.wav" in json.loads(result.stderr)["error"]

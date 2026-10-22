"""Sampling and distributed-batch contracts using tiny chronological PCM fixtures."""

import json
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.fabric.utilities.data import _set_sampler_epoch
from torch.utils.data import DataLoader

from src.data.wav_datamodule import (
    WAVManifestDataModule,
    WAVManifestDataset,
    _EpochDistributedSampler,
    _manifest_entries,
    validate_manifests,
)


def _wav(path, frames=512):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(struct.pack(f"<{frames}h", *range(frames)))
    return path


def _manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return str(path)


def _manifests(tmp_path, size=5):
    return {
        f"{split}_manifest": _manifest(
            tmp_path / f"{split}.jsonl",
            [{"path": _wav(tmp_path / f"{split}-{i}.wav").name} for i in range(size)],
        )
        for split in ("train", "val", "test")
    }


def test_segment_offsets_bound_chronological_reads(tmp_path, monkeypatch):
    audio = _wav(tmp_path / "audio.wav")
    rows = [{"path": audio.name, "offset_samples": 100, "segment_samples": 32}]
    manifest = _manifest(tmp_path / "data.jsonl", rows)
    calls = []
    original = wave.Wave_read.readframes

    def readframes(handle, frames):
        calls.append(frames)
        return original(handle, frames)

    monkeypatch.setattr(wave.Wave_read, "readframes", readframes)
    dataset = WAVManifestDataset(_manifest_entries(manifest), num_samples=16, random_crop=True)
    for epoch in range(5):
        samples = dataset[(0, epoch)]["waveform"] * 32768
        assert 100 <= samples[0] <= 116
        torch.testing.assert_close(samples, torch.arange(samples[0], samples[0] + 16))
    assert calls == [2, 16, 16, 16, 16, 16]


def test_audioset_ten_second_clip_supports_nine_real_chronological_windows(tmp_path):
    audio = tmp_path / "audioset.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x01\x00" * 160_000)
    dataset = WAVManifestDataset(
        [audio], num_samples=9 * 16160, require_full_length=True, random_crop=True
    )
    waveform = dataset[(0, 4)]["waveform"]
    assert waveform.shape == (145440,)
    assert torch.all(waveform == 1 / 32768)


def test_offset_only_defaults_to_remaining_recording_and_evaluation_is_fixed(tmp_path):
    audio = _wav(tmp_path / "audio.wav", frames=64)
    manifest = _manifest(tmp_path / "data.jsonl", [{"path": audio.name, "offset_samples": 48}])
    dataset = WAVManifestDataset(_manifest_entries(manifest), num_samples=8)
    assert dataset.segment_frame_counts == [16]
    torch.testing.assert_close(dataset[(0, 0)]["waveform"] * 32768, torch.arange(48, 56).float())
    assert torch.equal(dataset[(0, 0)]["waveform"], dataset[(0, 99)]["waveform"])


@pytest.mark.parametrize(
    "fields",
    [
        {"offset": 1},
        {"start_seconds": 1},
        {"segment_samples": None},
        {"offset_samples": -1},
        {"offset_samples": True},
        {"offset_samples": 1.2},
        {"segment_samples": 0},
        {"segment_samples": False},
        {"segment_samples": "16"},
    ],
)
def test_invalid_or_unknown_segment_fields_fail_with_line_context(tmp_path, fields):
    audio = _wav(tmp_path / "audio.wav")
    manifest = _manifest(tmp_path / "data.jsonl", [{"path": audio.name, **fields}])
    with pytest.raises(ValueError, match="data.jsonl:1"):
        _manifest_entries(manifest)


@pytest.mark.parametrize(
    "fields",
    [
        {"offset_samples": 512},
        {"offset_samples": 500, "segment_samples": 13},
    ],
)
def test_segment_must_be_inside_real_audio(tmp_path, fields):
    audio = _wav(tmp_path / "audio.wav")
    manifest = _manifest(tmp_path / "data.jsonl", [{"path": audio.name, **fields}])
    with pytest.raises(ValueError, match="outside|beyond"):
        WAVManifestDataset(_manifest_entries(manifest), num_samples=8)


def test_distinct_segments_allowed_in_split_but_recordings_cannot_cross_splits(tmp_path):
    manifests = _manifests(tmp_path, 1)
    source = tmp_path / "train-0.wav"
    rows = [
        {"path": source.name, "offset_samples": offset, "segment_samples": 64} for offset in (0, 64)
    ]
    _manifest(Path(manifests["train_manifest"]), rows)
    report = validate_manifests(**manifests, num_samples=32, require_full_length=True)
    assert report["splits"]["train"]["source_frames"] == 128
    _manifest(Path(manifests["val_manifest"]), [{"path": source.name, "offset_samples": 128}])
    with pytest.raises(ValueError, match="overlap across splits"):
        validate_manifests(**manifests)


def test_implicit_and_explicit_equal_segments_are_duplicates(tmp_path):
    audio = _wav(tmp_path / "audio.wav", frames=64)
    manifest = _manifest(
        tmp_path / "data.jsonl",
        [
            {"path": audio.name, "offset_samples": 16},
            {"path": audio.name, "offset_samples": 16, "segment_samples": 48},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate physical WAV segment"):
        WAVManifestDataset(_manifest_entries(manifest), num_samples=8)


def test_epoch_crops_reproduce_across_worker_counts_and_change_between_epochs(tmp_path):
    audio = [_wav(tmp_path / f"{i}.wav") for i in range(4)]
    dataset = WAVManifestDataset(audio, num_samples=16, random_crop=True, seed=42)
    first = [dataset[(i, 0)]["waveform"] for i in range(4)]
    second = [dataset[(i, 1)]["waveform"] for i in range(4)]
    assert any(not torch.equal(a, b) for a, b in zip(first, second, strict=True))
    sampler = _EpochDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=4, sampler=sampler, num_workers=1, persistent_workers=True
    )
    torch.testing.assert_close(next(iter(loader))["waveform"], torch.stack(first))
    _set_sampler_epoch(loader, 1)
    torch.testing.assert_close(next(iter(loader))["waveform"], torch.stack(second))
    _set_sampler_epoch(loader, 0)
    torch.testing.assert_close(next(iter(loader))["waveform"], torch.stack(first))


def test_datamodule_sets_epoch_in_custom_batch_sampler_and_keeps_eval_fixed(tmp_path):
    module = WAVManifestDataModule(
        **_manifests(tmp_path),
        num_samples=16,
        batch_size=2,
        minimum_batch_size=2,
        train_random_crop=True,
        seed=42,
    )
    module.setup()
    train, val = module.train_dataloader(), module.val_dataloader()
    _set_sampler_epoch(train, 3)
    assert all(epoch == 3 for batch in train.batch_sampler for _, epoch in batch)
    _set_sampler_epoch(val, 3)
    torch.testing.assert_close(next(iter(val))["waveform"][0] * 32768, torch.arange(16).float())


def test_two_rank_full_batches_are_balanced_disjoint_and_reproducible(tmp_path):
    manifests = _manifests(tmp_path, 17)
    ranks = []
    for rank in (0, 1):
        module = WAVManifestDataModule(
            **manifests,
            num_samples=16,
            batch_size=4,
            minimum_batch_size=2,
            train_drop_last=True,
            seed=42,
        )
        module.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        module.setup()
        loader = module.train_dataloader()
        _set_sampler_epoch(loader, 2)
        batches = list(loader.batch_sampler)
        assert len(loader) == len(batches) == 2
        assert all(len(batch) == 4 for batch in batches)
        assert all(epoch == 2 for batch in batches for _, epoch in batch)
        ranks.append([index for batch in batches for index, _ in batch])
        assert [batch["waveform"].shape[0] for batch in loader] == [4, 4]
    assert not set(ranks[0]) & set(ranks[1])
    assert len(set(ranks[0] + ranks[1])) == 16


def test_two_rank_merged_tails_have_identical_step_and_batch_sizes(tmp_path):
    manifests = _manifests(tmp_path, 9)
    sizes = []
    for rank in (0, 1):
        module = WAVManifestDataModule(
            **manifests, num_samples=16, batch_size=2, minimum_batch_size=2
        )
        module.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        module.setup()
        sizes.append([len(batch) for batch in module.train_dataloader().batch_sampler])
    assert sizes == [[2, 3], [2, 3]]


def test_preflight_matches_training_length_and_rank_minimum_checks(tmp_path):
    manifests = _manifests(tmp_path, 3)
    with pytest.raises(ValueError, match="padding cannot"):
        validate_manifests(**manifests, num_samples=1024, require_full_length=True)
    with pytest.raises(ValueError, match="requires at least 4"):
        validate_manifests(**manifests, num_samples=16, minimum_batch_size=2, world_size=2)
    with pytest.raises(ValueError, match="requires at least 8"):
        validate_manifests(
            **manifests, num_samples=16, batch_size=4, world_size=2, train_drop_last=True
        )


def test_experiment_preflight_uses_strict_length_with_explicit_overrides(tmp_path):
    manifests = _manifests(tmp_path, 2)
    args = [sys.executable, "-m", "src.validate_data", "--experiment", "upstream_hierarchical"]
    for split in ("train", "val", "test"):
        args.extend([f"--{split}", manifests[f"{split}_manifest"]])
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 1
    assert "padding cannot" in json.loads(result.stderr)["error"]
    result = subprocess.run(
        args
        + ["--num-samples", "16", "--batch-size", "2", "--world-size", "1", "--no-drop-last-train"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["require_full_length"] and report["minimum_batch_size"] == 2
    assert report["num_samples"] == 16


def test_h100_preflight_derives_world_size_and_requires_full_global_batch(tmp_path):
    manifests = _manifests(tmp_path, 2)
    args = [
        sys.executable,
        "-m",
        "src.validate_data",
        "--experiment",
        "audioset_h100",
        "--num-samples",
        "16",
    ]
    for split in ("train", "val", "test"):
        args.extend([f"--{split}", manifests[f"{split}_manifest"]])
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 1
    assert "world_size=2" in json.loads(result.stderr)["error"]
    assert "requires at least 64" in json.loads(result.stderr)["error"]
    result = subprocess.run(
        args + ["--world-size", "1", "--batch-size", "2"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["train_drop_last"] and report["require_full_length"]
    assert report["world_size"] == 1

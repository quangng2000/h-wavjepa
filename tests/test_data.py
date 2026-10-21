import json
import struct
import wave
from pathlib import Path

import pytest
import torch

from src.data.synthetic_datamodule import SyntheticAudioDataModule, SyntheticAudioDataset
from src.data.wav_datamodule import WAVManifestDataModule, WAVManifestDataset, _manifest_paths


def test_synthetic_pairs_have_exact_balance_and_same_events():
    dataset = SyntheticAudioDataset(size=10, num_samples=128)
    assert [int(dataset[i]["order_label"]) for i in range(len(dataset))] == [0, 1] * 5
    for index in range(0, len(dataset), 2):
        low_high, high_low = dataset[index], dataset[index + 1]
        assert torch.equal(low_high["waveform"][:64], high_low["waveform"][64:])
        assert torch.equal(low_high["waveform"][64:], high_low["waveform"][:64])
        assert torch.equal(low_high["detail_target"], high_low["detail_target"])


def test_synthetic_reproducibility_and_split_separation():
    first = SyntheticAudioDataset(size=8, seed=7)
    second = SyntheticAudioDataset(size=8, seed=7)
    changed = SyntheticAudioDataset(size=8, seed=8)
    val = SyntheticAudioDataset(size=8, seed=7, split="val")
    test = SyntheticAudioDataset(size=8, seed=7, split="test")
    for index in range(len(first)):
        for key in first[index]:
            assert torch.equal(first[index][key], second[index][key])
        assert not torch.equal(first[index]["waveform"], changed[index]["waveform"])
    streams = [
        {item[i]["waveform"].numpy().tobytes() for i in range(len(item))}
        for item in (first, val, test)
    ]
    assert not streams[0] & streams[1]
    assert not streams[0] & streams[2]
    assert not streams[1] & streams[2]


def test_synthetic_loaders_have_finite_float32_waveforms():
    module = SyntheticAudioDataModule(num_samples=128, batch_size=4)
    module.setup("fit")
    for loader in (module.train_dataloader(), module.val_dataloader(), module.test_dataloader()):
        batch = next(iter(loader))
        assert batch["waveform"].shape == (4, 128)
        assert batch["waveform"].dtype == torch.float32
        assert torch.isfinite(batch["waveform"]).all()
        assert batch["waveform"].abs().max() <= 1
        assert batch["order_label"].shape == (4,)
        assert batch["detail_target"].shape == (4, 3)


@pytest.mark.parametrize(
    "kwargs",
    [{"size": 3}, {"size": 0}, {"size": 2, "num_samples": 3}, {"size": 2, "sample_rate": 0}],
)
def test_synthetic_rejects_invalid_fixture_dimensions(kwargs):
    with pytest.raises(ValueError):
        SyntheticAudioDataset(**kwargs)


def _wav(path: Path, values=(0, 32767, -32768, 16384), sample_rate=16_000, channels=1):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(values)}h", *values))
    return path


def _manifest(path: Path, files: list[Path]):
    path.write_text("\n".join(json.dumps({"path": str(file)}) for file in files) + "\n")
    return str(path)


def test_wav_pcm_conversion_padding_and_crop(tmp_path):
    audio = _wav(tmp_path / "audio.wav")
    padded = WAVManifestDataset([audio], num_samples=6)[0]["waveform"]
    assert padded.dtype == torch.float32
    assert torch.equal(padded, torch.tensor([0, 32767 / 32768, -1, 0.5, 0, 0]))
    cropped = WAVManifestDataset([audio], num_samples=2)[0]["waveform"]
    assert torch.equal(cropped, padded[:2])


def test_wav_manifests_are_relative_to_manifest_and_load_batches(tmp_path):
    manifests = {}
    for split in ("train", "val", "test"):
        audio = _wav(tmp_path / f"{split}.wav")
        manifests[f"{split}_manifest"] = _manifest(tmp_path / f"{split}.jsonl", [Path(audio.name)])
    module = WAVManifestDataModule(**manifests, num_samples=6)
    module.setup()
    for loader in (module.train_dataloader(), module.val_dataloader(), module.test_dataloader()):
        batch = next(iter(loader))
        assert set(batch) == {"waveform"}
        assert batch["waveform"].shape == (1, 6)


def test_wav_rejects_duplicate_and_symlink_split_overlap(tmp_path):
    audio = _wav(tmp_path / "audio.wav")
    alias = tmp_path / "alias.wav"
    alias.symlink_to(audio)
    duplicate = _manifest(tmp_path / "duplicate.jsonl", [audio, alias])
    with pytest.raises(ValueError, match="Duplicate"):
        _manifest_paths(duplicate)
    train = _manifest(tmp_path / "train.jsonl", [audio])
    val = _manifest(tmp_path / "val.jsonl", [alias])
    test = _manifest(tmp_path / "test.jsonl", [_wav(tmp_path / "test.wav")])
    with pytest.raises(ValueError, match="overlap"):
        WAVManifestDataModule(train, val, test).setup()


def test_wav_rejects_empty_manifest_and_wrong_format(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="at least one"):
        _manifest_paths(empty)
    with pytest.raises(ValueError, match="sample rate"):
        WAVManifestDataset([_wav(tmp_path / "rate.wav", sample_rate=8_000)])
    with pytest.raises(ValueError, match="mono"):
        WAVManifestDataset([_wav(tmp_path / "stereo.wav", channels=2)])

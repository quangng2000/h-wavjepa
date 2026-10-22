import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest
import torch

from src.embed import extract_embeddings
from src.eval import evaluate
from src.inference import load_model
from src.train import train


def _wav_dataset(tmp_path):
    manifests = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack("<1024h", *[1000 * (i % 9) for i in range(1024)]))
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps({"path": path.name, "group_id": split}) + "\n")
        manifests[split] = manifest
    return manifests


@pytest.mark.integration
def test_real_wav_train_evaluate_and_extract(make_config, tmp_path):
    manifests = _wav_dataset(tmp_path)
    overrides = ["data=wav", *[f"data.{key}_manifest={path}" for key, path in manifests.items()]]
    # The smoke preset's synthetic size keys do not belong to the real data module.
    overrides += ["~data.train_size", "~data.val_size", "~data.test_size"]
    cfg = make_config(overrides)
    _, objects = train(cfg)
    checkpoint = objects["trainer"].checkpoint_callback.last_model_path
    rng_before = torch.random.get_rng_state().clone()
    loaded = load_model(checkpoint)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.hparams.input_sample_rate == 16000
    waveform = torch.linspace(-0.5, 0.5, 1024).reshape(1, -1)
    expected = objects["model"].eval().encode(waveform)
    actual = loaded.encode(waveform)
    for left, right in zip(expected["levels"], actual["levels"], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    eval_cfg = make_config(
        [*overrides, f"ckpt_path={checkpoint}", f"paths.output_dir={tmp_path / 'eval'}"],
        name="eval",
    )
    evaluate(eval_cfg)
    output = tmp_path / "embeddings"
    report = extract_embeddings(checkpoint, manifests["test"], output, num_samples=1024)
    assert report["status"] == "completed"
    assert report["examples"] == report["shards"] == 1
    shard = torch.load(output / "batch-000000.pt", weights_only=True)
    assert [list(level.shape) for level in shard["levels"]] == [[1, 32], [1, 48], [1, 64]]
    assert shard["protected"].shape == (1, 32)
    index = json.loads((output / "index.jsonl").read_text())
    assert index["row"] == 0
    with pytest.raises(FileExistsError):
        extract_embeddings(checkpoint, manifests["test"], output, num_samples=1024)
    command = [
        sys.executable,
        "-m",
        "src.embed",
        "--model-family",
        "legacy",
        "--checkpoint",
        checkpoint,
        "--manifest",
        str(manifests["test"]),
        "--output",
        str(tmp_path / "cli_embeddings"),
        "--num-samples",
        "1024",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "completed"


def test_loader_rejects_unversioned_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="missing H-WavJEPA"):
        load_model(checkpoint)


def test_loader_rejects_non_dictionary_checkpoint(tmp_path):
    checkpoint = tmp_path / "invalid.ckpt"
    torch.save([torch.ones(1)], checkpoint)
    with pytest.raises(ValueError, match="Lightning state dictionary"):
        load_model(checkpoint)

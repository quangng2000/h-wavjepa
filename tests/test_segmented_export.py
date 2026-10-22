"""Embedding export preserves manifest segments in both reads and output identity."""

import json
import wave

import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from src.embed import extract_embeddings


@pytest.mark.parametrize("family", ["legacy", "dynamics"])
def test_export_reads_requested_segments_and_records_offsets(make_config, tmp_path, family):
    experiment = "legacy/smoke" if family == "legacy" else "upstream_smoke"
    cfg = make_config([f"experiment={experiment}"])
    model = instantiate(cfg.model).eval()
    if family == "legacy":
        model.hparams.input_sample_rate = 16000
    samples = int(cfg.data.num_samples)
    checkpoint = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(checkpoint)
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(checkpoint, checkpoint_path)
    phase = np.arange(samples * 2, dtype=np.float64)
    recording = (9000 * np.sin(phase * phase / 2000)).astype("<i2")
    audio = tmp_path / "long.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(recording.tobytes())
    manifest = tmp_path / "segments.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"path": audio.name, "offset_samples": offset, "segment_samples": samples})
            + "\n"
            for offset in (0, samples)
        )
    )
    output = tmp_path / "embeddings"
    result = extract_embeddings(
        checkpoint_path,
        manifest,
        output,
        num_samples=samples,
        batch_size=2,
        model_family=family,
    )
    assert result["examples"] == 2
    index = [json.loads(row) for row in (output / "index.jsonl").read_text().splitlines()]
    assert [row["offset_samples"] for row in index] == [0, samples]
    assert [row["segment_samples"] for row in index] == [samples, samples]
    assert index[0]["path"] == index[1]["path"] == str(audio)
    shard = torch.load(output / "batch-000000.pt", weights_only=True)
    inputs = torch.from_numpy(recording.astype(np.float32) / 32768).reshape(2, samples)
    with torch.inference_mode():
        expected = (
            model.encode(inputs)["levels"]
            if family == "legacy"
            else model.encode(inputs, subsample=True)
        )
    for level, exported in zip(expected, shard["levels"], strict=True):
        torch.testing.assert_close(exported, level.mean(dim=1), atol=1e-7, rtol=1e-6)
    assert not torch.equal(shard["levels"][0][0], shard["levels"][0][1])

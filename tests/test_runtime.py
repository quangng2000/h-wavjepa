import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.train import train
from src.utils.runtime import atomic_json, file_sha256, validate_config


@pytest.mark.parametrize(
    "override, message",
    [
        ("data.num_samples=1025", "divisible"),
        ("data.num_samples=128", "at least two"),
        ("model.mask_group_size=3", "total encoder pool"),
        ("num_threads=0", "num_threads"),
        ("seed=-1", "seed"),
        ("data.num_samples=1024.5", "positive integer"),
    ],
)
def test_preflight_rejects_incompatible_configuration(make_config, override, message):
    cfg = make_config([override])
    with pytest.raises(ValueError, match=message):
        train(cfg)
    record = json.loads((Path(cfg.paths.output_dir) / "run.json").read_text())
    assert record["status"] == "failed"
    assert not (Path(cfg.paths.output_dir) / "checkpoints").exists()


def test_run_directory_cannot_silently_overwrite_another_run(make_config):
    cfg = make_config(["train=false", "test=false", "callbacks=none", "logger=[]"])
    train(cfg)
    path = Path(cfg.paths.output_dir) / "run.json"
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="choose a new"):
        train(cfg)
    assert path.read_bytes() == original
    assert json.loads(original)["status"] == "completed"


def test_atomic_json_preserves_previous_file_on_invalid_payload(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"status": "completed"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_audio_configuration_is_real_data_and_preflight_valid(make_config):
    cfg = make_config(["experiment=audio"])
    validate_config(cfg)
    assert cfg.data._target_.endswith("WAVManifestDataModule")
    assert OmegaConf.is_missing(cfg.data, "train_manifest")
    assert cfg.data.num_samples / cfg.data.sample_rate == 8


@pytest.mark.integration
def test_completed_run_records_metrics_and_checkpoint_hash(make_config):
    cfg = make_config()
    metrics, objects = train(cfg)
    output = Path(cfg.paths.output_dir)
    record = json.loads((output / "run.json").read_text())
    saved_metrics = json.loads((output / "metrics.json").read_text())
    assert record["status"] == "completed"
    assert record["global_step"] == 2
    assert record["device"] == "cpu"
    assert saved_metrics["val/loss"] == pytest.approx(metrics["val/loss"].item())
    checkpoint = objects["trainer"].checkpoint_callback.last_model_path
    assert record["checkpoints"]["last"]["sha256"] == file_sha256(checkpoint)

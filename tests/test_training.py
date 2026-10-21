import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.eval import evaluate
from src.train import train


@pytest.mark.integration
def test_fit_resume_and_evaluate(make_config, tmp_path):
    cfg = make_config()
    metrics, objects = train(cfg)
    model, trainer = objects["model"], objects["trainer"]
    assert objects["cfg"] is cfg
    assert {"datamodule", "callbacks", "logger"} <= objects.keys()
    assert trainer.global_step == 2
    assert model.ema_updates.item() == 2
    assert torch.isfinite(metrics["val/loss"])
    assert trainer.default_root_dir == cfg.paths.output_dir
    last = Path(trainer.checkpoint_callback.last_model_path)
    assert last.is_file()
    metadata = json.loads((Path(cfg.paths.output_dir) / "provenance.json").read_text())
    assert metadata["trainable_parameters"] > 0
    assert (Path(cfg.paths.output_dir) / "config.yaml").is_file()

    resumed = make_config(
        [
            "trainer.max_epochs=2",
            f"ckpt_path={last}",
            f"paths.output_dir={tmp_path / 'resumed'}",
        ]
    )
    _, resumed_objects = train(resumed)
    resumed_model, resumed_trainer = resumed_objects["model"], resumed_objects["trainer"]
    assert resumed_trainer.global_step == 4
    assert resumed_model.ema_updates.item() == 4

    eval_cfg = make_config(
        [
            f"ckpt_path={last}",
            f"paths.output_dir={tmp_path / 'evaluation'}",
        ],
        name="eval",
    )
    eval_metrics, eval_objects = evaluate(eval_cfg)
    evaluated = eval_objects["model"]
    assert torch.isfinite(eval_metrics["test/loss"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(evaluated.state_dict()[key], value)


def test_evaluation_requires_checkpoint(make_config):
    with pytest.raises(ValueError, match="requires ckpt_path"):
        evaluate(make_config(name="eval"))


def test_test_only_requires_explicit_checkpoint(make_config):
    with pytest.raises(ValueError, match="requires an explicit trusted ckpt_path"):
        train(make_config(["train=false", "test=true"]))


def test_training_rejects_missing_checkpoint(make_config, tmp_path):
    with pytest.raises(ValueError, match="existing trusted local checkpoint"):
        train(make_config([f"ckpt_path={tmp_path / 'missing.ckpt'}"]))


@pytest.mark.integration
def test_accumulation_updates_teacher_once_per_optimizer_step(make_config):
    cfg = make_config(["+trainer.accumulate_grad_batches=2"])
    _, objects = train(cfg)
    model, trainer = objects["model"], objects["trainer"]
    assert trainer.global_step == 1
    assert model.ema_updates.item() == 1


@pytest.mark.integration
def test_test_only_restores_checkpoint_without_fitting(make_config, monkeypatch, tmp_path):
    _, trained = train(make_config())
    checkpoint = trained["trainer"].checkpoint_callback.last_model_path

    def unexpected_fit(*args, **kwargs):
        pytest.fail("train=false must not call Trainer.fit")

    monkeypatch.setattr(type(trained["trainer"]), "fit", unexpected_fit)
    metrics, objects = train(
        make_config(
            [
                "train=false",
                "test=true",
                f"ckpt_path={checkpoint}",
                f"paths.output_dir={tmp_path / 'test_only'}",
            ]
        )
    )
    assert torch.isfinite(metrics["test/loss"])
    for key, value in trained["model"].state_dict().items():
        torch.testing.assert_close(objects["model"].state_dict()[key], value)


@pytest.mark.integration
def test_disabled_callbacks_and_logger(make_config):
    cfg = make_config(
        ["callbacks=none", "logger=[]", "+trainer.enable_checkpointing=false", "test=true"]
    )
    metrics, objects = train(cfg)
    assert objects["callbacks"] == []
    assert objects["logger"] == []
    assert objects["trainer"].logger is None
    assert objects["trainer"].checkpoint_callback is None
    assert torch.isfinite(metrics["test/loss"])


@pytest.mark.integration
def test_upstream_fast_dev_run(make_config):
    metrics, objects = train(make_config(["debug=fdr", "test=true"]))
    assert objects["trainer"].fast_dev_run
    assert objects["trainer"].global_step == 1
    assert objects["model"].ema_updates.item() == 1
    assert torch.isfinite(metrics["test/loss"])


def test_zero_seed_reproducibly_initializes_model_without_fit(make_config):
    overrides = ["seed=0", "train=false", "test=false", "callbacks=none", "logger=[]"]
    torch.manual_seed(123)
    first_metrics, first = train(make_config(overrides))
    torch.manual_seed(456)
    second_metrics, second = train(make_config(overrides))
    assert first_metrics == second_metrics == {}
    assert first["trainer"].global_step == second["trainer"].global_step == 0
    for key, value in first["model"].state_dict().items():
        torch.testing.assert_close(second["model"].state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.integration
def test_direct_script_bootstraps_root_and_runs_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "direct"
    environment = {
        key: value for key, value in os.environ.items() if key not in {"PROJECT_ROOT", "PYTHONPATH"}
    }
    result = subprocess.run(
        [
            sys.executable,
            str(root / "src" / "train.py"),
            "experiment=smoke",
            "seed=0",
            f"hydra.run.dir={output}",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "provenance.json").read_text())
    assert metadata["seed"] == 0
    assert (output / "checkpoints" / "last.ckpt").is_file()


@pytest.mark.integration
def test_cli_multirun_isolated_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--multirun",
            "experiment=smoke",
            "seed=1,2",
            f"paths.log_dir={tmp_path / 'logs'}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    paths = sorted(tmp_path.rglob("provenance.json"))
    records = [json.loads(path.read_text()) for path in paths]
    assert sorted(record["seed"] for record in records) == [1, 2]
    assert len({path.parent for path in paths}) == 2
    for path in paths:
        assert (path.parent / ".hydra" / "config.yaml").is_file()
        assert (path.parent / "checkpoints" / "last.ckpt").is_file()


@pytest.mark.integration
def test_optuna_sweep_returns_logged_metric(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "src/train.py",
            "--multirun",
            "experiment=smoke",
            "debug=fdr",
            "hparams_search=hwavjepa_optuna",
            "hydra.sweeper.n_trials=2",
            "hydra.sweeper.n_jobs=1",
            f"hydra.sweep.dir={tmp_path}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result_config = OmegaConf.load(tmp_path / "optimization_results.yaml")
    assert result_config.name == "optuna"
    assert result_config.best_value >= 0
    assert "model.optimizer.lr" in result_config.best_params
    for trial in ("0", "1"):
        run = tmp_path / trial
        config = OmegaConf.load(run / "config.yaml")
        assert config.optimized_metric == "val/loss"
        assert config.trainer.fast_dev_run is True
        assert (run / "provenance.json").is_file()

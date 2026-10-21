import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.eval import evaluate
from src.train import train


@pytest.mark.integration
def test_fit_resume_and_evaluate(make_config, tmp_path):
    cfg = make_config()
    model, trainer = train(cfg)
    assert trainer.global_step == 2
    assert model.ema_updates.item() == 2
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
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
    resumed_model, resumed_trainer = train(resumed)
    assert resumed_trainer.global_step == 4
    assert resumed_model.ema_updates.item() == 4

    eval_cfg = make_config(
        [
            f"ckpt_path={last}",
            f"paths.output_dir={tmp_path / 'evaluation'}",
        ],
        name="eval",
    )
    evaluated, eval_trainer = evaluate(eval_cfg)
    assert torch.isfinite(eval_trainer.callback_metrics["test/loss"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(evaluated.state_dict()[key], value)


def test_evaluation_requires_checkpoint(make_config):
    with pytest.raises(ValueError, match="requires ckpt_path"):
        evaluate(make_config(name="eval"))


@pytest.mark.integration
def test_accumulation_updates_teacher_once_per_optimizer_step(make_config):
    cfg = make_config(["+trainer.accumulate_grad_batches=2"])
    model, trainer = train(cfg)
    assert trainer.global_step == 1
    assert model.ema_updates.item() == 1


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
    records = [json.loads(path.read_text()) for path in tmp_path.rglob("provenance.json")]
    assert sorted(record["seed"] for record in records) == [1, 2]

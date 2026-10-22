# H-WavJEPA

Learn audio representations at three temporal scales with **PyTorch Lightning + Hydra**.

[![Tests](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml/badge.svg)](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](pyproject.toml)
[![Lightning–Hydra](https://img.shields.io/badge/template-Lightning--Hydra-792ee5)](https://github.com/ashleve/lightning-hydra-template)

[Quickstart](#quickstart) · [Train](#train-on-audio) · [Experiments](#experiments-with-hydra) · [Docs](#documentation)

<a id="proposed-three-space-architecture"></a>

![Three learned latent spaces with separate predictors, EMA targets, and per-level losses](docs/figures/h-wavjepa-three-latent-spaces.png)

*Proposed forecasting design; horizons and token counts are illustrative. The current research prototype uses [masked prediction](docs/ARCHITECTURE.md). No pretrained weights or benchmark gains are claimed.*

## Quickstart

Requires **Python 3.11–3.13** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/quangng2000/h-wavjepa.git
cd h-wavjepa
uv sync --frozen
uv run python -m src.train experiment=smoke
```

The smoke run uses generated audio on CPU. No dataset download or account needed.

## Train on audio

Prepare `train.jsonl`, `val.jsonl`, and `test.jsonl` with **mono, 16-bit PCM, 16 kHz WAV** files. Paths are relative to each manifest.

```json
{"path": "audio/example.wav", "group_id": "speaker-001"}
```

```bash
uv run python -m src.train experiment=audio \
  data.train_manifest=/data/train.jsonl \
  data.val_manifest=/data/val.jsonl \
  data.test_manifest=/data/test.jsonl
```

All three manifests are required. Keep recordings and speakers separate across splits. Clips are cropped or padded to eight seconds without resampling. See [data preparation](docs/USAGE.md#real-audio).

## Experiments with Hydra

Select a config, override values, or sweep seeds:

```bash
# Choose an experiment
uv run python -m src.train experiment=hwavjepa

# Override hyperparameters
uv run python -m src.train experiment=hwavjepa \
  trainer.max_epochs=20 model.optimizer.lr=0.0003

# Sweep seeds
uv run python -m src.train --multirun experiment=smoke seed=1,2,3
```

Configs live in [`configs/experiment/`](configs/experiment/). Configs, CSV metrics, and checkpoints are saved under `logs/train/runs/`; sweeps use `logs/train/multiruns/`.

## Project structure

```text
configs/       Hydra configs: data, models, trainers, experiments
src/
  data/        Datasets and Lightning DataModules
  models/      Encoders, predictors, and LightningModules
  objectives/  Training losses
  train.py     Training entry point
  eval.py      Evaluation entry point
tests/         Unit and integration tests
docs/          Guides and research notes
logs/          Generated run artifacts (gitignored)
```

## Documentation

- [Usage](docs/USAGE.md) — data, resume, evaluation, and embeddings
- [Architecture](docs/ARCHITECTURE.md) — hierarchy, masking, and objectives
- [Operations](docs/OPERATIONS.md) — checkpoints and hardware support
- [Theory](docs/THEORY_BOUNDARY.md) — assumptions and proof boundaries
- [Lightning–Hydra integration](docs/TEMPLATE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

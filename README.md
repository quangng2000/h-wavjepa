# H-WavJEPA

[![Tests](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml/badge.svg)](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml)
[![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc/)

Learn **three latent audio spaces at different prediction horizons** using the official [WavJEPA](https://github.com/labhamlet/wavjepa) backbone, [H-JEPA](https://github.com/kevinghst/H-JEPA) SIGReg, and the [Lightning–Hydra template](docs/TEMPLATE.md).

The model retains WavJEPA's masked objective and adds two learned abstractions and three future predictors. Released WavJEPA weights can initialize the backbone; the new hierarchy starts fresh. This is a research implementation with no demonstrated benchmark improvement yet.

![Three-space H-WavJEPA concept](docs/figures/h-wavjepa-three-latent-spaces.png)

*Conceptual sketch. The implementation uses completed 2.01-second windows, pointwise upper encoders, and configurable horizons. See the [architecture and exact temporal grids](docs/UPSTREAM_DYNAMICS.md).*

## Quick start

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra hear
uv run python -m src.train           # tiny official-backbone smoke run
uv run --extra hear pytest
```

The smoke run uses generated audio and random weights; it downloads no model or dataset.

## Train with pretrained WavJEPA

Create recording-disjoint `train.jsonl`, `val.jsonl`, and `test.jsonl` manifests:

```json
{"path": "audio/example.wav", "group_id": "speaker-001"}
```

```bash
uv run python -m src.train experiment=upstream_pretrained \
  data.train_manifest=/absolute/path/train.jsonl \
  data.val_manifest=/absolute/path/val.jsonl \
  data.test_manifest=/absolute/path/test.jsonl
```

This explicitly downloads the pinned, checksum-verified base weights (~785 MB), without executing Hub Python code. For offline use, set `model.backbone.pretrained=false model.backbone.pretrained_path=/path/model.safetensors`.

Inputs are mono PCM16 WAV at 16 kHz. The example preset requires **18.09 seconds of contiguous audio per file**, at least two files per split, and single-device FP32. Short training clips are rejected. CPU is tested; CUDA performance is unvalidated.

## Experiments and evaluation

| Experiment | Purpose |
| --- | --- |
| `upstream_pretrained` | Three spaces, initialized from released WavJEPA weights |
| `upstream_hierarchical` | Same architecture, random initialization |
| `upstream_flat` | Shared-space multi-horizon control |
| `upstream_baseline` | Original masked-loss adaptation control |

Use identical data, initialization and budgets for comparisons. Presets use warmup/cosine learning rates and annealed EMA. Runs and checkpoints live in `logs/train/runs/`; add `ckpt_path=/path/last.ckpt` to resume the same planned schedule.

- **Downstream quality:** [HEAR export and official evaluation](docs/HEAR.md), separately for low, medium, high, or backbone tokens.
- **Objective diagnostics:** `uv run python -m src.eval experiment=upstream_hierarchical ckpt_path=/path/last.ckpt` with the same data overrides.
- **Feature export:** `uv run python -m src.embed --model-family dynamics --checkpoint /path/last.ckpt --manifest /path/test.jsonl --output /new/output --num-samples 289440`.

## Project guide

```text
configs/             Hydra data, model, trainer and experiment presets
src/vendor/wavjepa/  Pinned official training core
src/models/          Audio hierarchy and forecasting integration
src/hear_api.py      HEAR embedding interface
evaluation/hear/     Isolated official evaluation runner
tests/              Model, causality, checkpoint and integration checks
```

[Design and training details](docs/UPSTREAM_DYNAMICS.md) · [Validation](docs/VALIDATION.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

The earlier independent model remains under explicit legacy experiments (`smoke`, `audio`, `hwavjepa`, `flat`); see [legacy usage](docs/USAGE.md). Its checkpoints remain distinct. Retained AudioMAE positional utilities carry their original noncommercial notice.

Development: `uv run ruff check .` and `uv run ruff format --check .`.

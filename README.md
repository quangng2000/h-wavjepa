# H-WavJEPA

[![Tests](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml/badge.svg)](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml)
[![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc/)

Three latent audio spaces built on the official [WavJEPA](https://github.com/labhamlet/wavjepa) training core, using [H-JEPA](https://github.com/kevinghst/H-JEPA)'s pointwise hierarchy and [LocaLeReg](docs/LOCALEREG.md). Experiments follow the [Lightning–Hydra template](docs/TEMPLATE.md).

The model retains masked prediction and adds three future predictors. Upper states transform individual windows; they **do not aggregate longer audio intervals**. This is a research baseline with no demonstrated benchmark improvement or measured two-H100 throughput yet.

![Three-space H-WavJEPA concept](docs/figures/h-wavjepa-three-latent-spaces.png)

*Conceptual sketch. The AudioSet preset uses 1.01-second windows; the long-recording preset uses 2.01 seconds. See [architecture and temporal grids](docs/UPSTREAM_DYNAMICS.md).*

## Quick start

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra hear
uv run python -m src.train              # tiny generated-audio smoke run
uv run --extra hear pytest
```

The smoke run downloads no data or weights. CPU tests cover model, gradient, checkpoint, sampling, and distributed behavior; GPU hardware validation is pending.

## Train on AudioSet

Prepare mono PCM16 WAV at 16 kHz and recording-disjoint JSONL manifests:

```json
{"path": "audio/example.wav", "recording_id": "source-recording-id"}
```

```bash
uv run python -m src.validate_data --experiment audioset_h100 \
  --train /data/audioset/train.jsonl \
  --val /data/audioset/val.jsonl --test /data/audioset/test.jsonl

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.train \
  experiment=audioset_h100 \
  data.train_manifest=/data/audioset/train.jsonl \
  data.val_manifest=/data/audioset/val.jsonl \
  data.test_manifest=/data/audioset/test.jsonl \
  hydra.run.dir=/checkpoints/hwavjepa/run-001
```

This selects DDP, BF16 mixed precision, gradient checkpointing, and 32 examples per GPU. Regularizers gather the **64-example global batch**; gradient accumulation does not enlarge that population. Training samples a new contiguous 9.09-second excerpt per epoch, so 10-second clips fit without padding. Memory capacity and throughput must be checked on the target GPUs before a full run.

[AudioSet guide](docs/AUDIOSET.md): manifest segments, pretrained initialization, exact timing, checkpoint resume, and evaluation. This adaptation uses the original 375,000-step budget, **not the original eight random two-second crop recipe**.

## Experiments

| Hydra experiment | Purpose |
| --- | --- |
| `audioset` | Pointwise hierarchy, random initialization, one-device FP32 |
| `audioset_h100` | Same AudioSet experiment configured for two H100s |
| `audioset_pretrained` | AudioSet hierarchy initialized from released WavJEPA weights |
| `audioset_flat` | Matched horizons and strides, upper maps replaced by identities |
| `upstream_hierarchical` / `upstream_pretrained` | Original 18.09-second long-recording presets |
| `upstream_sigreg` / `upstream_baseline` | SIGReg and masked-objective controls on long recordings |

For pretrained two-GPU training, add `model.backbone.pretrained=true` to `audioset_h100`. This explicitly downloads pinned, checksum-verified weights (~785 MB); only fixed sinusoidal positions are regenerated for the shorter window. Learned tensors load strictly. Use `model.regularizer_kind=sigreg` for a matched regularizer control.

Keep data, initialization, horizons, batch population, and budgets matched across comparisons. Resume with `ckpt_path=/path/last.ckpt` and a new run directory, retaining the original planned schedule.

## Evaluate and develop

- [HEAR](docs/HEAR.md): frozen scene/timestamp embeddings and official downstream evaluation.
- [AudioSet evaluation](docs/AUDIOSET.md#evaluation): objective diagnostics and embedding export; use one device for exact held-out coverage.
- [Validation evidence](docs/VALIDATION.md), [architecture](docs/UPSTREAM_DYNAMICS.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

```text
configs/             Hydra data, model, trainer and experiment presets
src/vendor/wavjepa/  Pinned official core plus documented compatibility changes
src/models/          Active waveform hierarchy and forecasting integration
src/legacy/          Archived prototype; old import paths remain compatible
src/hear_api.py      HEAR embedding interface
evaluation/hear/     Isolated official evaluation runner
tests/               Model, causality, data, distributed and resume checks
```

```bash
uv run ruff check .
uv run ruff format --check .
```

The [legacy prototype](src/legacy/README.md) remains accessible through explicit `smoke`, `audio`, `hwavjepa`, and `flat` experiments. Its checkpoint family is separate. Active sinusoidal positions use an independently implemented formula; dependency and checkpoint provenance remain documented in the notices.

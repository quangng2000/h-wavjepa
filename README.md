# H-WavJEPA

[![Tests](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml/badge.svg)](https://github.com/quangng2000/h-wavjepa/actions/workflows/tests.yml)
[![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd)](https://hydra.cc/)

Three latent audio spaces built on the official [WavJEPA](https://github.com/labhamlet/wavjepa) training core, using [H-JEPA](https://github.com/kevinghst/H-JEPA)'s pointwise hierarchy and [LocaLeReg](docs/LOCALEREG.md). Experiments follow the [Lightning–Hydra template](docs/TEMPLATE.md).

The model retains masked prediction and adds three future predictors. Upper states transform individual windows; they **do not aggregate longer audio intervals**. This is a research baseline with no demonstrated benchmark improvement yet.

![Three-space H-WavJEPA concept](docs/figures/h-wavjepa-three-latent-spaces.png)

*Conceptual sketch. The AudioSet preset uses 1.01-second windows; the long-recording preset uses 2.01 seconds. See [architecture and temporal grids](docs/UPSTREAM_DYNAMICS.md).*

## Quick start

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra hear --extra data
uv run python -m src.train              # tiny generated-audio smoke run
uv run --extra hear --extra data pytest
```

The smoke run downloads no data or weights. Tests cover model, gradient, checkpoint, sampling, and distributed behavior. A two-H100 BF16 train/resume test and three full-model generated-audio updates also passed; see [validation evidence](docs/VALIDATION.md).

## Train on AudioSet

Use the [CPU preparation guide](docs/PREPARE_AUDIOSET.md) to download the pinned WavJEPA AudioSet source, convert it to mono PCM16 WAV at 16 kHz, and build recording-disjoint JSONL manifests before renting GPUs. A [rolling S3 stager](docs/STAGE_AUDIOSET.md) prepares the full corpus on a smaller CPU disk:

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

This selects DDP, BF16 mixed precision, gradient checkpointing, and 32 examples per GPU. Regularizers gather the **64-example global batch**; gradient accumulation does not enlarge that population. Training samples a new contiguous 9.09-second excerpt per epoch, so 10-second clips fit without padding. Three full-model updates on two H100 80 GB GPUs used about **29.4 GiB peak allocated memory per GPU**. Sustained AudioSet throughput and total training time still require a real-data benchmark.

For a corpus stored as staged TAR shards, use the [streaming input guide](docs/SHARDED_AUDIO.md). It reads private S3 through WebDataset and a checksum-verified, bounded local cache; no full-corpus extraction is needed.

See the [performance guide](docs/PERFORMANCE.md) for token compaction, two/four-GPU comparisons, and reliable checkpoints with streaming input.

[AudioSet guide](docs/AUDIOSET.md): manifest segments, pretrained initialization, exact timing, checkpoint resume, and evaluation. This adaptation uses the original 375,000-step budget, **not the original eight random two-second crop recipe**.

## Experiments

| Hydra experiment | Purpose |
| --- | --- |
| `audioset` | Pointwise hierarchy, random initialization, one-device FP32 |
| `audioset_h100` | Same AudioSet experiment configured for two H100s |
| `audioset_h100_fast` / `audioset_h100_four` | Measured compact-token presets on two/four H100s, keeping global batch 64 |
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

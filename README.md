# H-WavJEPA

Train hierarchical audio representations at three temporal resolutions, with an optional protected information stream. Built with **PyTorch, Lightning, Hydra**, and the actual [Lightning–Hydra template](docs/TEMPLATE.md).

Includes real-WAV training, checkpoint resume, repeatable evaluation, and batch embedding extraction. CPU workflows are tested; useful downstream representations and accelerator performance still require experiments. This is not a pretrained model or an official WavJEPA reproduction.

## Proposed three-space architecture

![Proposed H-WavJEPA architecture with fine, intermediate, and coarse latent spaces, separate predictors, EMA target encoders, and a weighted sum of per-level losses](docs/figures/h-wavjepa-three-latent-spaces.png)

Conceptual forecasting design with three learned latent spaces. The 80 / 320 / 1,280 ms horizons and token counts are illustrative; see the [architecture documentation](docs/ARCHITECTURE.md) for the implemented objectives and their scope.

## Install and check

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run python src/train.py experiment=smoke
uv run pytest
```

The smoke run uses generated audio and requires no downloads or account.

## Train on audio

Create separate `train.jsonl`, `val.jsonl`, and `test.jsonl` manifests. Each line identifies a **mono, 16-bit PCM, 16 kHz WAV**. Paths are relative to the manifest; optional group IDs prevent known speakers/recordings from crossing splits.

```json
{"path": "audio/example.wav", "group_id": "speaker-001"}
```

```bash
uv run python -m src.validate_data \
  --train train.jsonl --val val.jsonl --test test.jsonl --num-samples 128000

uv run python src/train.py experiment=audio \
  data.train_manifest=/absolute/path/train.jsonl \
  data.val_manifest=/absolute/path/val.jsonl \
  data.test_manifest=/absolute/path/test.jsonl
```

The audio preset uses eight-second clips. All files are start-cropped or zero-padded; there is no automatic resampling. Set the epoch/batch budget for your dataset. Add `trainer=gpu` for a configured CUDA environment; see [validation limits](docs/OPERATIONS.md).

## Resume, evaluate, extract

Runs live in `logs/train/runs/`. Use the same experiment and data overrides when resuming or evaluating:

```bash
# Add your three data manifest overrides to these commands.
uv run python src/train.py experiment=audio ckpt_path=/absolute/path/last.ckpt
uv run python src/eval.py experiment=audio ckpt_path=/absolute/path/last.ckpt

# Architecture and sample rate are restored from the checkpoint.
uv run python -m src.embed \
  --checkpoint /absolute/path/last.ckpt --manifest test.jsonl \
  --output artifacts/embeddings --num-samples 128000
```

Each run records config, provenance, status, metrics, and checkpoints. Extraction writes mean-pooled vectors for every level into bounded-size PyTorch shards. Use new output directories and trusted checkpoints only.

## Documentation

- [Usage and experiment configuration](docs/USAGE.md)
- [Architecture and objective](docs/ARCHITECTURE.md)
- [Operations, checkpoint compatibility, and validation limits](docs/OPERATIONS.md)
- [Theory-to-code boundary](docs/THEORY_BOUNDARY.md) · [Lean proofs](https://github.com/quangng2000/hjepa-lean)
- [Upstream template integration](docs/TEMPLATE.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Development: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest`.

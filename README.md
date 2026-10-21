# H-WavJEPA

A Lightning + Hydra research scaffold for **low / medium / high temporal audio representations**, with an optional protected information stream.

**Status:** runnable small prototype, not a reproduction of released WavJEPA, not a trained foundation model, and not a Lean-verified neural implementation. Synthetic smoke checks validate software plumbing—not research hypotheses or superiority.

The organization follows [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), with independent project-specific code and configuration. See [template provenance](docs/TEMPLATE.md). Formal results live separately in [hjepa-lean](https://github.com/quangng2000/hjepa-lean).

## Quick start

From this project directory, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run pytest
uv run python -m src.train experiment=smoke
```

The smoke preset runs two CPU training batches and one validation batch on generated audio. It needs no dataset, account, pretrained checkpoint, or external logging service. It does **not** run a benchmark. Test-set evaluation is opt-in with `test=true`.

Runs are written to `logs/train/<date>/<time>/`:

- `config.yaml`: resolved experiment configuration.
- `provenance.json`: package versions, seed, Git revision/dirty status, parameter counts.
- `metrics/version_0/metrics.csv`: training/validation losses and diagnostics.
- `checkpoints/`: best validation-loss checkpoint and `last.ckpt`.
- `.hydra/`: Hydra configuration and override history.

## Structure

```text
configs/
  data/          Synthetic fixture or explicitly split local WAV manifests
  model/         Architecture, protection, optimizer, teacher and masking
  experiment/    Flat, two-level, three-level and protection ablations
  trainer/       CPU, Apple MPS and CUDA presets
  callbacks/     Checkpoints and learning-rate logging
  logger/        Local CSV logging
  paths/         Data and output locations
  hydra/         Single-run and sweep directories
  local/         Ignored machine-specific overrides
src/
  data/          Datasets and LightningDataModules
  models/        Lightning orchestration and independent model components
  objectives/    Loss functions
  utils/         Object construction and run provenance
  train.py       Training / checkpoint resume entry point
  eval.py        Explicit checkpoint evaluation entry point
tests/           Data, masking, EMA, configuration and checkpoint tests
docs/            Design decisions and theory boundaries
```

## What the model does

The predictive path patchifies audio without overlapping windows, embeds each patch, and builds progressively coarser features by aligned mean pooling plus learned projections. Each level has its own position-aware bidirectional Transformer predictor. A frozen EMA copy of the encoder supplies targets. Hidden waveform patches are removed **before** student encoding; a shared raw-patch mask is propagated through the hierarchy. This is noncausal masked prediction, not forecasting.

The protection path is independent: it learns to encode and reconstruct **unmasked raw patches**, which serve as a genuinely fixed reference. Its fine-resolution representation is carried unchanged in every level's readout bundle. It never feeds masked predictors. Reconstruction of raw audio is expensive and may preserve nuisance information; it is a transparent initial reference, not a claim of semantic abstraction.

The objective is a weighted sum of masked per-level latent MSE and optional raw-reference reconstruction MSE. Raw JEPA losses from independently learned spaces are **not** comparable downstream task scores. Monitor collapse even when predictive loss decreases; EMA is not an anti-collapse proof.

The default 1,024-sample / 16 kHz fixture is only **64 ms long**. With 32-sample patches and pooling factors `[2, 2]`, local encoder token supports are 2 / 4 / 8 ms. Predictors can attend to all permitted tokens in the clip. These tiny settings are for tests, not the proposed acoustic / event / phrase time scales. “High” refers to hierarchy level and slower temporal structure, not frequency or loudness. For larger contexts, ensure `num_samples` is divisible by `patch_size * mask_group_size`, and make `mask_group_size` compatible with the cumulative pooling factors.

## Configurable comparisons

```bash
# Three-level protected prototype (default)
uv run python -m src.train experiment=hwavjepa

# Isolate protection from hierarchy
uv run python -m src.train experiment=hwavjepa_unprotected
uv run python -m src.train experiment=flat
uv run python -m src.train experiment=flat_protected
uv run python -m src.train experiment=two_level

# Hydra sweep: separate directories for every run
uv run python -m src.train --multirun experiment=smoke seed=1,2,3

# Hardware choice: only CPU is part of the automated smoke check
uv run python -m src.train experiment=smoke trainer=mps
uv run python -m src.train experiment=smoke trainer=gpu
```

These presets share the fixture/context and raw masking group, but are **not parameter- or compute-matched**. Parameter counts are recorded; a matched-budget study and shared-space multiscale baseline remain to implement. MPS/CUDA configs are provided, not claims of hardware validation. No TPU or distributed validation is claimed.

## Your own WAV files

Use three JSONL manifests, one per split. Each line contains a local path:

```json
{"path": "recordings/example.wav"}
```

Relative paths resolve against the manifest's directory. Files must already be **16-bit PCM, mono, 16 kHz** (or the configured sample rate). No automatic resampling or data download occurs. The loader uses a deterministic start crop or zero padding to `num_samples`. It rejects duplicate resolved paths, including symlink aliases across splits.

```bash
uv run python -m src.train data=wav \
  data.train_manifest=/absolute/path/train.jsonl \
  data.val_manifest=/absolute/path/val.jsonl \
  data.test_manifest=/absolute/path/test.jsonl \
  data.num_samples=128000 model.encoder.patch_size=320 model.protected_dim=320
```

Choose patch widths, pooling, batch size and memory budget before large-context runs; the command only illustrates the input interface. Split by **original recordings, speakers/performers, and generating factors** as appropriate—not by overlapping clips. Path checks cannot detect copied audio, shared subjects, or statistical leakage. All three manifests are required so the split check is explicit, even when `test=false`.

## Resume and evaluate

```bash
uv run python -m src.train experiment=smoke \
  ckpt_path=/absolute/path/last.ckpt trainer.max_epochs=2

uv run python -m src.eval experiment=smoke \
  ckpt_path=/absolute/path/last.ckpt
```

Use the **same architecture, data and objective overrides** recorded in the checkpoint run's `config.yaml`. A checkpoint does not automatically select a Hydra experiment. Lightning checkpoints include model/EMA weights, optimizer state, and training progress. Exact bitwise continuation of random sampling is not guaranteed here. Load only checkpoints you trust.

`src.eval` currently measures held-out objective terms. It is **not** HEAR evaluation, a trained downstream probe, or a generalization certificate. Synthetic order/detail labels are fixture metadata and are not used by the self-supervised objective.

## Theory boundary and next milestones

The Lean project proves finite conditional guarantees, explicit failure examples, and convergence of specified small linear optimizers. It does not prove that AdamW trains this encoder, that EMA avoids collapse, or that this hierarchy improves real audio tasks. See [theory-to-code obligations](docs/THEORY_BOUNDARY.md).

Next research work:

1. Reproduce the official [WavJEPA](https://github.com/labhamlet/wavjepa) checkpoint/evaluation pipeline independently of this prototype.
2. Lock a recording/factor-disjoint benchmark, downstream probes, and matched context/compute budgets.
3. Add shared-space multiscale controls, all-level versus upper-only probes, variance/effective-rank analysis, ablations, and uncertainty estimates.
4. Run experiments only after the protocol and resource budget are chosen; report negative results too.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

No remote repository or training service is required. Data, checkpoints and run outputs are ignored by Git. `uv.lock` pins the environment; CI runs the same CPU checks.

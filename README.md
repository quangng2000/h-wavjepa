# H-WavJEPA

A Lightning + Hydra research scaffold for **low / medium / high temporal audio representations**, with an optional protected information stream.

**Status:** runnable small prototype, not a reproduction of released WavJEPA, not a trained foundation model, and not a Lean-verified neural implementation. Synthetic smoke checks validate software plumbing—not research hypotheses or superiority.

Built on the **actual [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)**, imported at commit `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144`. Its train/eval pipeline, task wrapper, utilities, configuration groups, and root discovery are retained, with MNIST replaced by our H-WavJEPA components. See the [import and customization ledger](docs/TEMPLATE.md). Formal results live separately in [hjepa-lean](https://github.com/quangng2000/hjepa-lean).

## Quick start

From this project directory, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run pytest
uv run python src/train.py experiment=smoke
```

The smoke preset runs two CPU training batches and one validation batch on generated audio. It needs no dataset, account, pretrained checkpoint, or external logging service. It does **not** run a benchmark. Test-set evaluation is opt-in with `test=true`.

The template's direct-script entry points and `python -m src.train` / `src.eval` both work. `rootutils` discovers `.project-root`, sets `PROJECT_ROOT`, and loads an optional ignored `.env` file.

Runs are written to `logs/train/runs/<date>_<time>/`:

- `config.yaml`: resolved experiment configuration.
- `provenance.json`: package versions, seed, Git revision/dirty status, parameter counts.
- `csv/version_0/metrics.csv`: training/validation losses and diagnostics.
- `checkpoints/`: best validation-loss checkpoint and `last.ckpt`.
- `.hydra/`: Hydra configuration and override history.
- `config_tree.log` / `tags.log`: the upstream Rich configuration and tag helpers.

## Structure

```text
configs/
  data/          Synthetic fixture or explicitly split local WAV manifests
  model/         Architecture, protection, optimizer, teacher and masking
  experiment/    Flat, two-level, three-level and protection ablations
  trainer/       CPU, Apple MPS and CUDA presets
  callbacks/     Checkpoints, early stopping and Rich progress/summary
  logger/        CSV plus upstream optional logger presets
  debug/         Fast-dev, limited-batch, overfit and profiler presets
  extras/        Rich config printing and tag handling
  hparams_search/ Optuna search configuration
  paths/         Data and output locations
  hydra/         Single-run and sweep directories
  local/         Ignored machine-specific overrides
src/
  data/          Datasets and LightningDataModules
  models/        Lightning orchestration and independent model components
  objectives/    Loss functions
  utils/         Upstream instantiation, logging, Rich output and task wrapper
  train.py       Training / checkpoint resume entry point
  eval.py        Explicit checkpoint evaluation entry point
tests/           Data, masking, EMA, configuration and checkpoint tests
docs/            Design decisions and theory boundaries
notebooks/       Upstream notebook workspace
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
uv run python src/train.py --multirun experiment=smoke seed=1,2,3

# Upstream debug and Optuna integration (pipeline checks, not research evidence)
uv run python src/train.py experiment=smoke debug=fdr
uv run python src/train.py -m experiment=smoke debug=fdr \
  hparams_search=hwavjepa_optuna hydra.sweeper.n_trials=2

# Disable loggers / explicitly configured callbacks
uv run python src/train.py experiment=smoke 'logger=[]' callbacks=none

# Hardware choice: only CPU is part of the automated smoke check
uv run python -m src.train experiment=smoke trainer=mps
uv run python -m src.train experiment=smoke trainer=gpu
```

These presets share the fixture/context and raw masking group, but are **not parameter- or compute-matched**. Parameter counts are recorded; a matched-budget study and shared-space multiscale baseline remain to implement. MPS/CUDA configs are provided, not claims of hardware validation. No TPU or distributed validation is claimed.

Optuna minimizes validation objective only; it is not a downstream-performance comparison. The regular HPO preset allows 20 trials: set an explicit budget before using it beyond smoke checks. Upstream W&B, TensorBoard, MLflow, Comet, Neptune, and Aim configuration files are included but their optional SDKs are not installed or activated by default. Install/configure only the integration you choose. Those external services and the inherited DDP presets are unvalidated here.

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

The upstream Python API returns `(metric_dict, object_dict)` from both `train(cfg)` and `evaluate(cfg)`; retrieve the model/trainer as `object_dict["model"]` / `object_dict["trainer"]`. `train=false test=true` requires an explicit checkpoint so an evaluation cannot silently use newly initialized weights.

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
uv run pre-commit run --all-files
```

No remote training service is required. Data, checkpoints and run outputs are ignored by Git. `uv.lock` pins the environment; CI runs the CPU checks. `requirements.txt` also exposes the runtime dependencies for pip users, while `uv` remains the reproducible installation path. Pre-commit hooks are available but are not automatically installed into your Git hooks.

Upstream template attribution and its MIT notice are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

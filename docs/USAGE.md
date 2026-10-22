# Archived prototype usage

**Historical, frozen research route.** Its implementation lives in
[`src/legacy`](../src/legacy/README.md); existing imports, checkpoints and experiment
names remain compatibility interfaces. New experiments should use AudioSet/upstream
presets. This page covers the earlier independent prototype. For the new official-WavJEPA
training extension, use [Upstream dynamics](UPSTREAM_DYNAMICS.md).

Run commands from the repository after `uv sync --frozen`. Direct-script train/eval entry points also discover the checkout from another directory. Hydra configs execute Python construction instructions; use trusted configurations only.

## Real audio

```bash
uv run python -m src.validate_data \
  --train /data/train.jsonl --val /data/val.jsonl --test /data/test.jsonl \
  --sample-rate 16000 --num-samples 128000

uv run python src/train.py experiment=legacy/audio trainer=cpu \
  data.train_manifest=/data/train.jsonl \
  data.val_manifest=/data/val.jsonl \
  data.test_manifest=/data/test.jsonl \
  trainer.max_epochs=20 data.batch_size=8
```

All three manifests are required, even when test evaluation is disabled. Each JSONL record has a nonempty `path`; relative paths resolve against the manifest directory. WAVs must already be uncompressed 16-bit PCM, mono, and at the configured sample rate. Training samples a random contiguous crop from longer segments; validation and extraction start at the manifest offset. Short segments are right zero-padded in this legacy route. Padded samples contribute to loss: consistently prepare clips for meaningful experiments.

Optional `group_id` and `recording_id` enforce known group separation. If a field appears anywhere, it must appear on **every** record across all splits. The same ID may repeat inside a split, not across splits. Choose appropriate independent units such as speaker, performer or original recording. Multiple declared segments of one recording may appear within one split. Optional `offset_samples` and `segment_samples` delimit each segment at the configured sample rate. Duplicate physical segments and any recording shared across splits, including symlink/hardlink aliases, are rejected.

Validation checks headers and declared payload ends without decoding/hashing the entire corpus. Reports include file counts and padded/cropped clip counts. Copied recordings, overlapping crops and incorrect labels are not detectable by these checks. Keep files immutable during runs.

### Audio preset

| Setting | Value |
| --- | --- |
| Clip | 128,000 samples / 8 seconds at 16 kHz |
| Patch | 320 samples / 20 ms |
| Hierarchy widths | 128 / 192 / 256 |
| Pool factors | 4 / 4 |
| Token counts | 400 / 100 / 25 |
| Encoder token supports | 20 / 80 / 320 ms |
| Protected width | 128 |

Each predictor can attend over visible clip context. These are configurable starting settings, not a validated optimum or a guarantee of acoustic/event/phrase semantics. CPU is the default; `trainer=gpu` and `trainer=mps` require appropriate environments and workload validation.

Standard loader settings are exposed: `data.num_workers`, `pin_memory`, `persistent_workers`, and `prefetch_factor`. Persistent workers and explicit prefetching require positive worker count. To debug the audio preset without workers, use `data.num_workers=0 data.persistent_workers=false data.prefetch_factor=null`.

## Comparisons and sweeps

```bash
uv run python src/train.py experiment=legacy/hwavjepa
uv run python src/train.py experiment=legacy/hwavjepa_unprotected
uv run python src/train.py experiment=legacy/flat
uv run python src/train.py experiment=legacy/flat_protected
uv run python src/train.py experiment=legacy/two_level

uv run python src/train.py -m experiment=legacy/smoke seed=1,2,3
uv run python src/train.py experiment=legacy/smoke debug=fdr
uv run python src/train.py -m experiment=legacy/smoke debug=fdr \
  hparams_search=hwavjepa_optuna hydra.sweeper.n_trials=2
```

Comparison presets use small synthetic fixtures unless `data=wav` and manifests are supplied. They share masking groups but are **not parameter/compute matched**. Synthetic labels are fixture metadata, not supervision. The full Optuna preset permits 20 trials; specify a budget. Validation objective is not a cross-model downstream score.

`num_samples` must be divisible by `patch_size * mask_group_size`, contain at least two groups, and mask groups must align with the cumulative pooling factor. Preflight checks these before model allocation.

## Resume and evaluate

```bash
# Include the same model/data overrides used for the original run.
uv run python src/train.py experiment=legacy/audio \
  data.train_manifest=/data/train.jsonl data.val_manifest=/data/val.jsonl \
  data.test_manifest=/data/test.jsonl \
  ckpt_path=/runs/previous/checkpoints/last.ckpt trainer.max_epochs=40

uv run python src/eval.py experiment=legacy/audio \
  data.train_manifest=/data/train.jsonl data.val_manifest=/data/val.jsonl \
  data.test_manifest=/data/test.jsonl ckpt_path=/runs/previous/checkpoints/last.ckpt
```

`max_epochs` is the total target, not additional epochs. Lightning restores model/EMA weights, optimizer and loop state; a different current `lr` override does not replace restored optimizer settings. Exact random-stream/dataset-position continuation is not promised. Always use a new output directory, including on resume.

Lightning resets some ModelCheckpoint ranking state when its directory changes. Treat the new run's best checkpoint as its local best, and retain the original run/checkpoint for comparison; the runner does not maintain a global best across run directories.

Architecture, objective, masking, EMA, protection and sample rate must match the checkpoint contract. Legacy unversioned checkpoints are rejected; see [operations](OPERATIONS.md). Validation/test masks are repeatable for the same seed **and batch/shard layout**. Changing batch size, ordering or distributed layout can change scores.

`src.eval` measures held-out objective, not downstream accuracy. `train=false test=true` requires a checkpoint. After fitting, `test=true` uses the best checkpoint when available, otherwise current weights with a warning. Do not repeatedly tune on the test set.

## Extract features

```bash
uv run python -m src.embed --model-family legacy --checkpoint /runs/previous/checkpoints/last.ckpt \
  --manifest /data/test.jsonl --output artifacts/test-embeddings \
  --num-samples 128000 --batch-size 16 --device cpu
```

Architecture and sample rate come from checkpoint metadata, without executing saved Hydra YAML. Clip length is explicit; use the training length for comparable features. The full training checkpoint is loaded into host memory, then waveform batches/features stream in bounded-size shards. This is not a low-memory checkpoint loader or serving system.

- `metadata.json`: status, hashes, sample rate, clip length, widths, shard count. Only `completed` indicates success.
- `index.jsonl`: original path, manifest offset/segment length, shard filename and row per input.
- `batch-000000.pt`, etc.: `levels`, a list of `[batch, width]` tensors, plus optional `protected`. Read with `torch.load(path, weights_only=True)`.

Extraction mean-pools temporal tokens. Pooling can discard information; the protected-stream theory does not certify these vectors. For full tokens:

```python
from src.inference import load_model

model = load_model("/runs/previous/checkpoints/last.ckpt", device="cpu")
features = model.encode(waveform)  # float [batch, samples], already at the saved sample rate
levels = features["levels"]  # one [batch, tokens, width] tensor per level
protected = features["protected"]
```

No trained classifier or audio generator is implied.

## Logging and development

CSV logging is local by default. Disable it with `'logger=[]'`; disable explicit callbacks with `callbacks=none`. Optional W&B, TensorBoard, MLflow, Comet, Neptune and Aim presets require separate SDKs/accounts. Put credentials in SDK environment/key stores, not in Hydra configs, which are saved and may be logged.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

`uv.lock` is the reproducible environment. `requirements.txt` lists runtime dependencies but is not a lockfile. The repository retains the template's checkout-based layout, not an independently packaged wheel.

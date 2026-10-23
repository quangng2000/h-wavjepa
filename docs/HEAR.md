# HEAR evaluation

H-WavJEPA implements the [HEAR API](https://hearbenchmark.com/hear-api.html) in
`src.hear_api`. It evaluates a saved **upstream-backed dynamics checkpoint**;
the legacy masked prototype is a different model family. No model downloads or
remote Python imports occur during evaluation.

Feature extraction uses the training environment. Label alignment, embedding
packaging, shallow probe training, validation selection, and metrics use the
[official HEAR evaluation kit](https://github.com/hearbenchmark/hear-eval-kit),
pinned to `855964977238e89dfc76394aa11c37010edb6f20`, in a separate environment.
This avoids mixing its Lightning 1.x API with the training project's Lightning 2.x.

## Setup

From the repository root:

```bash
uv sync --extra hear
uv venv --python 3.11 artifacts/hear-env
uv pip install --python artifacts/hear-env/bin/python --torch-backend cpu \
  -r evaluation/hear/requirements.txt
uv pip install --python artifacts/hear-env/bin/python --no-deps \
  'heareval @ git+https://github.com/hearbenchmark/hear-eval-kit.git@855964977238e89dfc76394aa11c37010edb6f20'
```

The explicit dependency recipe replaces upstream's obsolete NumPy/Numba pins.
`--no-deps` is intentional. The runner applies two checksum-verified compatibility
patches: optional TensorFlow for PyTorch-only embedding packaging, and the
Lightning 1.9 epoch counter used to retrieve event postprocessing selected at the
best validation epoch. It does not change
metrics, probes, split logic, or validation selection. Only the local generated
probe checkpoints and locally exported embeddings enter this legacy environment.
CPU evaluation is tested; CUDA is not validated here.

## Extract and evaluate

Use a task prepared in the [official HEAR dataset format](https://github.com/hearbenchmark/hear-preprocess):
`task_metadata.json`, `labelvocabulary.csv`, split JSON files, and mono WAV files
under `<sample_rate>/<split>/`. Prepare the required sample rate beforehand;
the exporter rejects stereo and mismatched rates instead of silently converting.
Task labels, splits, and evaluation parameters are preserved.

```bash
uv run --frozen --extra hear python -m src.hear_eval \
  --checkpoint logs/train/runs/<run>/checkpoints/last.ckpt \
  --task-dir data/hear/<task> \
  --output-dir artifacts/hear/high/<task> \
  --representation high --device cpu --window-batch-size 8

OMP_NUM_THREADS=1 artifacts/hear-env/bin/python evaluation/hear/run.py \
  artifacts/hear/high/<task> --grid default --grid-points 8 --workers 0
```

Use a fresh output directory for each checkpoint, representation, and experiment.
Extraction refuses to overwrite existing outputs. The official runner skips a
completed prediction directory; use a new export for a new evaluation protocol.
`--prepare-only` runs the official label alignment and memmap packaging without
training a probe. The evaluator writes its normal `test.predicted-scores.json`
and `prediction-done.json`, plus upstream provenance. Export provenance records
the checkpoint SHA-256, representation, audio-window settings, saved
`readout_mode`, configured `average_top_k_layers`, and `hierarchy` flag.
`representation_readout` distinguishes the chosen window readout from the
`backbone` representation's final-layer tokens. With a flat checkpoint,
`duplicate_levels` explicitly lists `low`, `medium`, and `high`: these are
identical features and must not be reported as three distinct learned spaces.

## Representation and timing

For pointwise checkpoints:

| `--representation` | Scene feature | Timestamp feature |
| --- | --- | --- |
| `low` | Duration-weighted mean of window states | One state per completed window |
| `medium` | Same pooling after learned G2 | Dense G2 states on the same window grid |
| `high` | Same pooling after learned G2 then G3 | Dense G3 states on the same window grid |
| `backbone` | Mean of contextual frontend-grid tokens | Actual backbone tokens on their native grid |

Every level has the backbone's feature width. For pointwise checkpoints, dense upper-level readouts apply
the learned pointwise maps to every window; they do not upsample sparse states.
This evaluation grid differs from the default training state strides `[1, 2, 4]`;
both are recorded in the export manifest.
Window geometry and the low-state readout come from the checkpoint. The original
32,160-sample window spans 2.01 seconds at 16 kHz; AudioSet checkpoints use
16,160 samples, or 1.01 seconds. New checkpoints use normalized top-k low states;
older checkpoints preserve their original final-layer readout. Both window-level
grids are too coarse for precise event timing. Use `backbone` for that
comparison: timestamps come from the configured convolution geometry, and
slightly overlapping context windows maintain a continuous token grid. With the
default frontend the token hop is 10 ms. This is a token location, **not** a
10 ms context window or a streaming availability guarantee: contextual attention
and normalization see the complete independently encoded audio window.

For [temporal checkpoints](TEMPORAL_AGGREGATION.md), medium/high states aggregate
past low states using the saved geometry. Only complete supports are emitted on
the dense grid, and timestamp locations are the centers of those full supports.
With the default 1/2/4-window geometry, a nine-window clip has 9/8/6 dense states.
Feature extraction retains past low states across window batches, so changing
`window_batch_size` does not reset the temporal context. Low states are cached on
CPU; accelerator chunks include only the needed lookback and current outputs.

When a clip has fewer windows than the selected upper support, HEAR repeats its
first low state on the left and emits one state at the real clip center. This is
an explicit **evaluation-only fallback**; training and ordinary `encode` never
pad state history. Timestamp extraction retains its existing rule of omitting a
partial waveform tail unless the clip is shorter than one base window. Scene
extraction retains that tail, pools valid dense states and weights them by each
endpoint window's real duration. Overlapping supports are not individually
duration-weighted. Report this convention with scene scores.

Export metadata includes `aggregation_windows`, `level_support_windows`,
`selected_support_windows`, `scene_pooling`, and `short_clip_state_padding`.
Identity upper maps with temporal pooling are not reported as duplicate levels.

Scene readouts retain a trailing partial window, right-pad it with zeros, and
weight it by its real duration. Coarse timestamp readouts omit a partial trailing
window. If a clip is shorter than one window, they return one padded-context
feature at the real clip center. Backbone readouts omit tokens whose frontend
support crosses the end of real audio; below one receptive field they return
one padded token at the clip center. Padding can affect contextual representations
and should remain identical across compared models.

The API receives `[sounds, samples]` float32 mono audio in `[-1, 1]`, as specified
by HEAR. It cannot infer the true lengths of already padded inputs. The exporter
therefore processes each file separately and batches only its fixed-size windows.
It does not treat silence as padding or trim it. Outputs are float32 CPU tensors;
only `window_batch_size` windows are moved to the accelerator at a time.

## Tests

```bash
uv run --frozen --extra hear pytest tests/test_hear.py -q
uv run --frozen --extra hear python -m evaluation.hear.make_smoke \
  --output artifacts/hear-smoke
OMP_NUM_THREADS=1 artifacts/hear-env/bin/python evaluation/hear/run.py \
  artifacts/hear-smoke/export-scene artifacts/hear-smoke/export-event \
  --grid-points 1 --workers 0
```

The smoke creates disjoint synthetic clips and a tiny randomly initialized
checkpoint, then exercises the official scene and event evaluators. Its two-epoch
task-specific grid is only an integration test, not a benchmark protocol or a
representation-quality result. Real comparisons need the same held-out task
splits, evaluation grid, and data preparation across baseline, flat, and
hierarchical checkpoints, with results reported separately for each readout.
Regression tests use multiple encoder layers and a nonidentity final LayerNorm
to distinguish normalized top-k from legacy readouts, including checkpoints
saved before the readout-mode field existed.

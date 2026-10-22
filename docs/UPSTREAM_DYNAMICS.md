# Hierarchical dynamics on official WavJEPA

This route extends the actual [WavJEPA training core](https://github.com/labhamlet/wavjepa)
at commit `6f4e1b1f18316c50768c8523f325ccffeaa6985b`. It does not rebuild its
convolutional frontend, context Transformer, masked predictor, or masked loss.
The source subset, hashes, compatibility edits and licenses are recorded in
[`src/vendor/wavjepa/NOTICE.md`](../src/vendor/wavjepa/NOTICE.md).

## Research references and adaptation boundary

| Mechanism | Evidence | This implementation |
| --- | --- | --- |
| Waveform features, masked context/target prediction, EMA context encoder | [WavJEPA, Section 3.1](https://arxiv.org/html/2509.23238v1#S3.SS1), [pinned training source](https://github.com/labhamlet/wavjepa/tree/6f4e1b1f18316c50768c8523f325ccffeaa6985b) | Reuses the official core and masked loss. |
| Compositional latent spaces at coarser temporal strides | [H-JEPA, Section 2.1, Eq. 1](https://arxiv.org/html/2610.06805v1#S2.SS1), [pinned three-level AntMaze configuration](https://github.com/kevinghst/H-JEPA/blob/97054ecaff83e13839195ec4b92d74c08a0127c5/h_jepa/config/train/ant_hjepa_l3.yaml) | Pointwise upper MLPs with state window size 1; default relative strides 2 then 2. |
| Per-space future prediction and SIGReg | [H-JEPA, Section 2.1, Eqs. 2–3](https://arxiv.org/html/2610.06805v1#S2.SS1), [pinned loss implementation](https://github.com/kevinghst/H-JEPA/blob/97054ecaff83e13839195ec4b92d74c08a0127c5/h_jepa/loss.py) | Separate audio predictors and the attributed SIGReg implementation. |

H-JEPA's experiments use **state window size 1**. The general formulation allows
larger windows, but this implementation follows the experimental pointwise choice.
The two upper transitions are:

```text
medium[t] = G2(low[2*t])
high[t]   = G3(medium[2*t])
```

Here `state_strides: [1, 2, 4]` measures cumulative strides relative to low states;
it corresponds to H-JEPA's per-transition strides `[2, 2]`. Each higher stride must
be an integer multiple of the preceding one. State counts follow the input length:
`floor((N - 1) / stride) + 1`, with no fixed 8/4/2 token requirement. Training
subsamples **before** each upper MLP. The same positions are used for forecasting
and regularization. Each upper state consumes one lower state, not an average of
adjacent states.

Our completed-audio-window mean readout, five-window shared history, single
forecast origin, action-free predictors, and EMA upper targets are **audio
adaptation choices**, not H-JEPA's published recipe. Its action aggregation has
no counterpart in these action-free recordings. The mean readout pools WavJEPA
tokens *inside one audio window*; it is separate from the pointwise hierarchy
*between windows*. Keeping WavJEPA's masked objective and extending EMA to upper
targets makes this a hybrid of the two references, not a reproduction of either
paper's complete training experiment.

## What changes

The original masked objective remains active. Two small pointwise MLPs create
nested state spaces above the trainable WavJEPA encoder. Each level has a causal
PyTorch Transformer forecasting head; upper-level losses update both MLPs and
the shared encoder. A `hierarchy=false` control replaces the MLPs with identity
maps while keeping the same predictor architecture, horizons, and temporal strides.

For completed window index `t`, low states are mean readouts of contextual
WavJEPA tokens, medium states are `G2(low)`, and high states are `G3(medium)`.
This mean readout is an explicit design choice that can discard within-window
details; it is not evidence of learned semantic abstraction. The original
masked objective still trains the finer token-level features.

The default history contains five consecutive windows. Every predictor's origin
is window 4 (zero-based), and the target indices are 5, 6, and 8:

| Level | Observed window indices | Target index | Horizon |
| --- | --- | --- | --- |
| Low | 0, 1, 2, 3, 4 | 5 | 1 window |
| Medium | 0, 2, 4 | 6 | 2 windows |
| High | 0, 4 | 8 | 4 windows |

Horizons and strides are configurable examples, not experimentally established
optimal scales. A stride does **not** average the intervening windows: G2/G3
are pointwise, and each target describes one future window rather than the entire
interval up to it. All levels have the same width but distinct learned coordinates.

The loss is original masked MSE plus weighted own-space future MSE plus SIGReg.
SIGReg is reused from [H-JEPA](https://github.com/kevinghst/H-JEPA), commit
`97054ecaff83e13839195ec4b92d74c08a0127c5`, under MIT. It is evaluated across
recordings separately at each observed timestep, not by treating a recording's
time steps as independent examples. The coefficient and projection count are
configurable. It penalizes collapsed upper states; it does not prove preservation
of useful information, successful optimization, or superiority over a flat model.

Future targets use WavJEPA's existing EMA context Transformer and EMA copies of
G2/G3, with stop-gradient targets. The frontend/norm/mapper remain shared and
detached on the target path, as in upstream WavJEPA. Teacher top-layer averaging
and normalization are retained. EMA updates only after successful optimizer
steps. This is **not** H-JEPA's no-EMA, action-conditioned training recipe.

## Chronology and causality

Each input row is one contiguous recording excerpt, reshaped into nine ordered,
nonoverlapping windows. Each window is normalized and encoded independently.
The forecasting encoder only receives the first five windows; it cannot access
future normalization statistics or convolutional support. The separate original
masked branch can use every training window but does not feed its activations
to forecasting heads. Never join shuffled audio crops into fake trajectories.

The base preset uses explicit 32,160-sample windows at 16 kHz (2.01 seconds).
The adapter passes an integer length rather than converting through seconds;
upstream's floating-point conversion otherwise truncates this to 32,159. Both
give the released model's 200-token positional grid. The pinned convolutional
frontend has a computed stride of 160 samples (10 ms) and receptive field of
240 samples (15 ms), before its full-window contextual Transformer.
History is 10.05 seconds; target window endpoints are 2.01, 4.02 and 8.04 seconds
beyond the history endpoint. Total required recording length is **18.09 seconds**.
A nominal 10-second AudioSet clip is too short for this preset. Short WAVs are
rejected, never zero-padded. This route is not token-by-token streaming; all
features of a bidirectional window become available only when that window ends.

## Run

```bash
uv sync --frozen
# Actual upstream code at tiny dimensions, random weights and generated audio:
uv run python -m src.train experiment=upstream_smoke
```

For real data, supply recording-disjoint JSONL splits of mono PCM16 16 kHz WAVs:

```bash
uv run python -m src.train experiment=upstream_hierarchical \
  data.train_manifest=/absolute/path/train.jsonl \
  data.val_manifest=/absolute/path/val.jsonl \
  data.test_manifest=/absolute/path/test.jsonl
```

The `upstream_hierarchical` preset initializes **randomly**. Choose
`experiment=upstream_pretrained` with the same data overrides to initialize the
backbone from the official Hugging Face base release. Only the new abstractions
and future predictors start fresh. The loader pins revision
`6be4a5093f6c9adbbd55e00b6e5b8f067aa03345`, verifies the complete safetensors
SHA-256, removes the explicit `model.` prefix, and requires exact tensor names,
shapes and dtypes. It does not execute Hub Python or its configuration expressions.

```bash
# Optional: cache the ~785 MB file before training; prints its local path.
uv run python -m src.pretrained
# Confirm it is cached without networking:
uv run python -m src.pretrained --local-files-only
```

`model.backbone.pretrained_path=/absolute/path/model.safetensors` accepts an
explicit local export without networking (set `pretrained=false`). Its layout
and finite tensors are validated and its digest recorded; local custom exports
are not claimed to be the released weights. Alternatively,
`model.backbone.checkpoint_path=/absolute/path/wavjepa-state.pt` accepts a trusted
tensor-only original training checkpoint with `weights_only=True`. Initialization
sources are mutually exclusive. A full `ckpt_path` resume/evaluation bypasses
initialization downloads/files because Lightning restores all model weights.

The actual released weights were loaded and compared numerically against a
separate hash-verified original source checkout; see [validation](VALIDATION.md).

Use `upstream_flat` and `upstream_baseline` with the same data/budget overrides.
The baseline disables future losses and SIGReg. It retains our chronological
window sampling, FP32 execution and optimizer settings; it is an **adaptation
control**, not a reproduction of the published 375k-step training run. The flat
control removes G2/G3 parameters, so it is not exactly parameter-matched; report
the parameter/compute difference and tune a capacity-matched control for claims.

Initial support is single-device FP32. CPU is tested; CUDA performance is not
validated. DDP, mixed precision, gradient checkpointing, and compilation are
rejected. The presets reuse Transformers' cosine-with-warmup scheduler and
WavJEPA's linear EMA annealing, controlled by the outer Lightning module:

```yaml
model:
  lr_schedule: cosine
  lr_warmup_steps: null   # floor(10% of the resolved optimizer-step budget)
  lr_total_steps: null    # Trainer.estimated_stepping_batches
  ema_decay: 0.999
  ema_end_decay: 0.99999
  ema_anneal_end_step: null  # same resolved budget
```

These budget-scaled defaults are an adaptation, not the original paper's
100k-step warmup/EMA horizon. Set explicit values for a planned research run.
The resolved schedule is checkpointed; resuming preserves its budget, including
with gradient accumulation. Continuing beyond a completed cosine budget is
rejected. Choose the full `lr_total_steps` before an interrupted/partial run.
Older checkpoints without schedule fields retain constant LR/EMA semantics.

SIGReg requires at
least two recordings per batch; the real-data presets enforce this, merging a
singleton final batch into the preceding batch without dropping held-out records.
Each split must contain at least two recordings. Fixed batch layout is required
for repeatable SIGReg evaluation; a changed last-batch size changes its statistic.

Resume/evaluate with the same model and data configuration:

```bash
uv run python -m src.train experiment=upstream_smoke ckpt_path=/absolute/path/last.ckpt
uv run python -m src.eval experiment=upstream_smoke ckpt_path=/absolute/path/last.ckpt
```

This model family has its own checkpoint schema. Select it explicitly for
bounded, mean-pooled embedding export on each level's training grid:

```bash
uv run python -m src.embed --model-family dynamics \
  --checkpoint /absolute/path/last.ckpt --manifest /absolute/path/test.jsonl \
  --output /new/embeddings --num-samples 289440
```

For scene/timestamp benchmark features and official scoring, use [HEAR](HEAR.md).
The Python interface keeps dense and subsampled readouts explicit:

```python
import torch
from src.dynamics_inference import load_model

model = load_model("/absolute/path/last.ckpt")
# audio: float32 [batch, complete_windows * model.window_samples], correct sample rate
with torch.no_grad():
    levels = model.encode(audio)  # each [batch, complete_windows, feature_width]
    temporal_levels = model.encode(audio, subsample=True)  # each level's coarser grid
    predictions = model.predict(audio[:, : model.context_windows * model.window_samples])
```

`encode` keeps its dense default for feature-extraction compatibility. Use
`subsample=True` for the hierarchy's actual temporal grids; `encode_targets`
supports the same option for EMA states. Training returns these subsampled online
states in `compute_loss(...)["levels"]`. Future targets are selected by their
original window indices from dense teacher states, which is equivalent to
pointwise encoding only the selected timestamps.

Predictions are latent vectors, not synthesized audio. Current training uses one
origin per excerpt; it does not establish stable recursive rollouts or planning.

## Evidence and limits

Independent checks compare original-source outputs/gradients and the attributed
SIGReg formula. Tests also check actual upstream execution, gradients into the backbone and both
abstractions, future-input invariance, horizon indexing, collapsed-state penalty,
teacher freezing/timing, checkpoint round trips, and Lightning resume. They do
not establish useful representations or reproduce published scores. The source
comparison can be repeated with a separate pinned upstream checkout:

```bash
uv run python -m src.verify_upstream --upstream /path/to/original/wavjepa
# Requires the explicitly downloaded official weights in cache:
uv run python -m src.verify_upstream --upstream /path/to/original/wavjepa --pretrained
```

Do not compare own-space losses across different learned encoders as evidence of
quality: the target representations can change scale or difficulty. Compare
shared downstream speech/music/environmental tasks, matched context and budgets,
and a common fixed-reference forecasting metric. Include static/persistence
controls and changing-event subsets; predictable recording identity is not useful
dynamics. There are no actions or planning claims and no new Lean guarantee.

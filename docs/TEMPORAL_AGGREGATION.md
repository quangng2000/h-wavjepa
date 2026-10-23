# Temporal aggregation and within-recording pressure

`audioset_temporal` adds causal temporal aggregation and an optional upper-space
variance floor. The existing `audioset` family remains the pointwise baseline.
These are new audio experiments, not reproduced H-JEPA results or a demonstrated
improvement. Both retain the original WavJEPA masked objective and the same
nine-window AudioSet recipe.

## State support and targets

Let `low[t]` be the normalized top-k window readout. With cumulative strides
`[1,2,4]` and `aggregation_windows: [1,2,2]`, the dense states are:

```text
medium[t] = G2(mean(low[t-1], low[t]))                 t >= 1
high[t]   = G3(mean(medium[t-2], medium[t]))           t >= 3
```

G2 and G3 retain the same MLP widths as the pointwise arm. Each high state sees
two adjacent two-window blocks, covering four consecutive low states. No learned
absolute time index enters these state encoders. Predictor position embeddings
remain separate. Student and EMA targets use the same aggregation geometry.

| Space | Audio support per state | Observed prediction endpoints | Future target support |
| --- | --- | --- | --- |
| Low | 1 window = 1.01 s | 0, 1, 2, 3, 4 | 5 |
| Medium | 2 windows = 2.02 s | 2, 4 | 5–6 |
| High | 4 windows = 4.04 s | 4 | 5–8 |

All indices are zero-based window indices. The forecasting origin remains the end
of window 4. Targets end at 5, 6 and 8, so every target support lies after the
observed prefix. The masked branch may train on all nine windows, but its future
activations do not enter online forecasting states or the temporal penalty.

Only fully supported states are used: training neither pads missing history nor
averages a shorter startup window. The dense valid grids for five observed windows
are low `0..4`, medium `1..4`, and high `3..4`. Forecasting selects endpoints on the
original stride grids. Cross-recording SIGReg/LocaLeReg uses those forecasting
grids; the temporal penalty and temporal variance diagnostics use the dense valid
grids. This distinction is saved in configuration and extraction metadata.

The high predictor consequently has **one context state**, containing four low
windows. Its context-order permutation test cannot measure ordering sensitivity.
Its dense variance estimate has only two overlapping states, so it is noisy and
should not be treated as a calibrated semantic score. Increasing the observed
history requires a separately matched data recipe.

For other configurations, support widths follow
`R[0]=1; R[l]=R[l-1]+(aggregation_windows[l]-1)*state_strides[l-1]`.
The first valid endpoint is `R[l]-1`. Dense encoding retains every valid endpoint;
subsampled encoding retains endpoints divisible by that level's stride. General
settings can leave gaps inside the support envelope; the preset above has
contiguous support. Configurations without sufficient observed support are rejected.

## Upper-space temporal variance floor

For each recording `b` and coordinate `d`, compute population variance over the
**dense valid observed states** at each upper level, independently of other
recordings:

```text
std[l,b,d] = sqrt(var_over_time(z[l,b,:,d], correction=0) + epsilon)
L_temporal = sum(l=1,2) mean_over_b_d(relu(gamma - std[l,b,d]))
L_total = L_existing + temporal_variance_weight * L_temporal
```

The temporal presets initially use weight `0.1`, floor `gamma=0.1` and
`epsilon=1e-4`. These are explicit starting hyperparameters, not tuned values.
Statistics run in FP32. Every upper space needs at least two valid context states.
DDP averages this per-recording penalty normally; it does not mix recordings into
the time axis or require a feature queue. The cross-recording regularizer retains
its existing global-batch implementation.

A recording-constant upper state incurs `gamma-sqrt(epsilon)=0.09` per level,
even if its values vary greatly across recordings. Thus zero forecasting error
and a diverse marginal distribution no longer make that state cost-free. However,
the smooth variance statistic has zero gradient at exact equality; this is not a
guarantee of escape from every collapsed initialization. Noise, channel variation
or unhelpful feature oscillation can satisfy the floor. Truly stationary audio can
also be penalized. Tune on held-out data and report downstream scene and event
results, persistence and absolute within/between-recording variance together.

Masking does not provide a theorem against collapse of a jointly learned teacher:
it discourages constant predictions when the token targets retain variation.
Likewise, action-conditioned predictors can ignore their actions; H-JEPA uses
additional inverse-dynamics supervision on diverse scenes ([Section 4.3](https://arxiv.org/html/2610.06805v1#S4.SS3)).
The temporal floor here is an explicit audio-side constraint on upper states,
not action supervision or a reproduction of that mechanism.

## Joint-time constraints and next ablations

[H-JEPA Appendix L](https://arxiv.org/html/2610.06805v1#A12) constructs an
episode-constant encoder satisfying a marginal distribution constraint when an
observable episode-specific background provides sufficient variation. Conditioning
on actions alone does not exclude that solution. Its diverse-scene experiments
use an inverse-dynamics objective with action labels; the tested label-free time
repulsion did not recover successful planning. AudioSet's scene diversity makes
this a relevant risk, but it does not establish that the released audio readout
is already collapsed. Measure that with the [frozen probe](TEMPORAL_DIAGNOSTICS.md#frozen-released-weight-probe).

LocaLeReg here clusters recordings separately at each time; it supplies no
within-recording joint-time constraint. EMA is not such a constraint either. The
variance floor does couple a recording's states across time, but it remains a
soft, experimental penalty with the limitations above. Aggregation adds temporal
support without by itself excluding a recording-constant solution.

Two **unimplemented** pairwise objectives are candidates for separate ablations:

- **Time-offset classification:** predict the signed time difference from two
  upper states, without passing indices, timestamps or predictor positional
  embeddings to the auxiliary head. Sample multiple balanced offsets within
  each recording. A constant state pair cannot distinguish those labels. A
  single positive offset supplies no such pressure; in this five-window preset,
  the high level has only two dense valid states, so multiple positive distances
  require longer history. Both directions supply a narrower direction-only task.
- **Detached low-level change prediction:** regress `stop_grad(low[t2]-low[t1])`
  from the corresponding upper-state pair. This resembles inverse dynamics but
  uses a learned audio change as a pseudo-action. Targets must vary conditional
  on recording identity: one fixed pair per recording permits memorization of
  its change from identity alone. Multiple pairs or both directions help test
  that issue. Stop-gradient prevents direct target chasing in this loss; it does
  not make an uninformative low representation informative.

For squared-error change prediction, target variance alone is insufficient:
unpredictable zero-mean changes still make a zero-delta predictor optimal. Compare
against zero-delta and recording-conditioned mean controls, and test whether
the upper pair explains predictable within-recording change.

For either candidate, a residual loss at constant states is a useful diagnostic,
not proof that optimization escapes collapse or preserves useful abstraction.
Report conditional target variation, constant-pair reference loss, persistence,
and downstream quality. A head can reward nuisance variation, and timestamp
leakage can solve an offset task without meaningful audio dynamics.

## Matched experiments

Run the four cells independently with the same data index, seeds, initialization,
optimizer schedule, batch size and evaluation protocol:

| Arm | Four-H100 Hydra overrides |
| --- | --- |
| Pointwise baseline | `experiment=audioset_h100_four` |
| Pointwise + temporal pressure | `experiment=audioset_h100_four model.temporal_variance_weight=0.1` |
| Aggregation only | `experiment=audioset_temporal_h100_four model.temporal_variance_weight=0` |
| Aggregation + temporal pressure | `experiment=audioset_temporal_h100_four` |

The matched masked-only reference remains `audioset_baseline_h100_four`. The
temporal family also provides `audioset_temporal` (CPU), `_h100` (two GPUs with
activation checkpointing), and `_h100_fast` (two GPUs with compact masking).
Their execution presets inherit existing settings; GPU throughput for the new
model has not been measured. Equal updates do not imply equal FLOPs.

```bash
# Generated-audio CPU integration run; no downloads or GPU rental.
uv run python -m src.train experiment=upstream_temporal_smoke

# Four-H100 aggregation arm, after supplying the same data overrides as the baseline.
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 -m src.train \
  experiment=audioset_temporal_h100_four \
  data.train_manifest=/data/audioset/train.jsonl \
  data.val_manifest=/data/audioset/val.jsonl \
  data.test_manifest=/data/audioset/test.jsonl \
  hydra.run.dir=/checkpoints/hwavjepa/temporal-001
```

The pointwise and aggregate arms use different numbers of valid states. Their
regularizer populations, temporal overlap and target supports therefore differ
even though the audio excerpt is identical. Report those differences. Separate
capacity, readout and upper-target-EMA ablations remain necessary for narrower
causal claims.

## Checkpoints and evaluation

Aggregation geometry, temporal weight, floor and epsilon are part of the saved
model contract. Resuming under changed values is rejected. Older checkpoints
without these fields retain pointwise, zero-temporal-penalty semantics. Loading
weights for a new experiment is a separate initialization operation.

Dense `encode` outputs contain only states with complete support, so levels can
have different lengths. Use `level_positions` and `level_support_windows` to
interpret them; do not infer time indices from tensor length. HEAR and embedding
export retain the saved geometry and report their evaluation padding and timing
conventions. See [HEAR](HEAR.md) for short clips and scene pooling.

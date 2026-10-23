# Pointwise baseline and temporal diagnostics

The original hierarchy remains a **pointwise baseline**. G2 and G3 transform a
single completed audio-window state. Subsampling at strides 2 and 4 does not give
those states a longer receptive field. The separately configured
[temporal version](TEMPORAL_AGGREGATION.md) adds genuine aggregation and an optional
within-recording training penalty. The diagnostics on this page remain read-only.

A state can vary across recordings but stay constant throughout each recording.
This may satisfy the cross-recording distribution penalty and make forecasting
easy without learning within-recording dynamics. Neither low prediction loss nor
a nonzero batch standard deviation rules this out.

Action inputs would not automatically remove this solution: a predictor can
ignore them. H-JEPA addresses slow-feature collapse on diverse scenes with an
additional inverse-dynamics loss ([Section 4.3](https://arxiv.org/html/2610.06805v1#S4.SS3)).
This action-free audio adaptation has no equivalent supervision. Its EMA targets
and marginal regularizer are not a guarantee against the recording-identity shortcut.

Validation and test now log these measurements for every level:

| Metric suffix | Meaning |
| --- | --- |
| `within_recording_variance` | Mean population variance across context times, then recordings and features. |
| `between_recording_variance` | Population variance across each recording's context mean, averaged over features. |
| `within_variance_fraction` | Within variance divided by within + between; zero when both vanish. Always inspect the absolute variances too. |
| `persistence_mse` | Error from repeating the EMA state of the final observed window as the future target. |
| `forecast_gain_over_persistence` | Persistence MSE minus trained predictor MSE; positive means the predictor improved on persistence. |
| `time_shifted_context_mse` | Predictor error after cyclically shifting context states by one position while retaining their position labels. |
| `time_shifted_context_mse_increase` | Shifted-context MSE minus ordinary prediction MSE. |
| `variance_context_states` | Number of dense valid states used for within-recording variance. |
| `predictor_context_states` | Number of strided states supplied to the predictor/order probe. |

For example, `val/level_2/within_recording_variance` measures the dense valid high
context grid: five states for the pointwise model, two for the default temporal
model. Small within variance alongside larger
between variance identifies a recording-constant risk. Low persistence error and
no cost from disrupting context order provide further evidence, but these are
diagnostics, not a learned semantic-quality score or a prevention mechanism.

Aligned Lightning evaluation estimates variance **within each global validation
batch**, gathering latent states across ranks before computing nonlinear ratios.
`val/diagnostic_recordings` records the population size. Standalone evaluation
uses its supplied population; older distributed reports used local batches and
are not directly comparable. Averages over batches are not whole-corpus variance.
Keep global batch size and sample grouping/order fixed for comparisons. The context
grids have different supports, overlap and valid counts, so compare a level
against matched controls, not absolute values across grids. Variance diagnostics
now use dense valid states for both model versions; older pointwise reports used
the sparse training grids and are not directly comparable. Predictor-order probes
still use strided states. The high temporal predictor has only one context state,
so its cyclic shift is unchanged and cannot diagnose order sensitivity. The cyclic
shift is deterministic and tests one ordering disruption, not all permutations.
Synthetic degeneracy tests check that recording-constant states remain diverse
between recordings yet have zero temporal variance and perfect persistence.

Persistence uses teacher-space origin and future states, avoiding a false penalty
from student/EMA drift. The default masked branch has already encoded the origin,
so these diagnostics reuse its teacher activations and add only small predictor
passes. With the masked objective disabled, validation encodes one additional
teacher origin window. They run on validation/test, not training steps, and do not
change the training loss.

## Frozen released-weight probe

Measure the level-zero readout before training upper maps:

```bash
uv run python -m src.probe_readout \
  --weights /data/weights/wavjepa/model.safetensors \
  --manifest /data/audioset/train.jsonl \
  --output /data/reports/wavjepa-frozen-readout-128.json \
  --max-recordings 128 --batch-size 4 --window-batch-size 8 --device cpu
```

Use `--device cuda` on existing CUDA hardware. Inputs must already exist locally:
the command checks the pinned official checkpoint SHA-256, freezes the online
encoder, and uses normalized top-eight-layer features averaged over token
positions. It downloads nothing and trains no hierarchy. The manifest must use
the [WAV manifest format](AUDIOSET.md#data-and-manifests); audio must be mono 16 kHz and long
enough for nine contiguous 16,160-sample windows. Short audio is rejected.

The seeded cohort takes one fixed manifest segment per recording, grouping both
recording IDs and physical-file aliases. It honors manifest offsets, with no
random crop, resampling or padding. Encoder memory is bounded by
`--window-batch-size`, independently of the recording batch size. The JSON records
checkpoint and manifest hashes, selected entries, offsets, geometry, seed and
readout. It refuses to overwrite an existing report. Audio payload hashes are
not checked; the supplied manifest remains responsible for dataset provenance.

`statistics` describes all nine windows; `observed_prefix_statistics` describes
the first five states from the same forward passes, matching the default training
prefix. Both include absolute within/between-recording variance, their fraction,
feature norms and adjacent-window change. Unlike the training diagnostics above,
the between term covers the **whole selected cohort**, using merged float64
moments, and is independent of batch partitioning up to rounding.

Between-recording variance dominating the within term would support investigating
the readout and upper-space shortcut. It is not a calibrated collapse threshold,
and feature scale matters. The probe's tests use constructed states and tiny
fixtures; no released-weight AudioSet result is claimed in this repository yet.

## Flat-control regularization

New configurations save `regularization_scope: unique_spaces`. The hierarchical
model keeps three penalties, one for each distinct distribution on its original
context grid. The flat model applies **one penalty to the full low-level context
grid**. Its three forecasting heads, horizons, strides, and prediction weights are
unchanged; the upper identities receive no duplicate penalties.

Dividing the old flat sum by three would not produce this objective: the stride-2
and stride-4 losses average different overlapping timesteps and random sketches.
Regularizing the full low grid once matches the hierarchy's backbone-space
penalty. The hierarchical arm still has additional penalties for its learned upper
spaces, so this comparison measures a hierarchy *with its upper-space collapse
protection*, not the architecture alone at an identical total regularizer count.
Report this distinction with results.

Old checkpoints without the new field resolve to `per_level` and retain their
original objective. Resuming them under `unique_spaces` is rejected as a changed
research configuration. Loading old weights for a new experiment must be explicit;
it is not a silent objective change on resume.

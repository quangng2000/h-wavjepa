# Pointwise baseline and temporal diagnostics

The current hierarchy remains a **pointwise baseline**. G2 and G3 transform a
single completed audio-window state. Subsampling at strides 2 and 4 does not give
those states a longer receptive field. A future aggregation model needs a
separate experiment and matched controls.

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

For example, `val/level_2/within_recording_variance` measures the high-level context
grid, which currently has only two states. Small within variance alongside larger
between variance identifies a recording-constant risk. Low persistence error and
no cost from disrupting context order provide further evidence, but these are
diagnostics, not a learned semantic-quality score or a prevention mechanism.

Variance is estimated **within each local validation batch** and then averaged
over batches/ranks; the between term is not whole-corpus variance. The context
grids have different numbers and spacings of observations, so compare a level
against its matched flat control, not absolute values across grids. The cyclic
shift is deterministic and tests one ordering disruption, not all permutations.
Synthetic degeneracy tests check that recording-constant states remain diverse
between recordings yet have zero temporal variance and perfect persistence.

Persistence uses teacher-space origin and future states, avoiding a false penalty
from student/EMA drift. The default masked branch has already encoded the origin,
so these diagnostics reuse its teacher activations and add only small predictor
passes. With the masked objective disabled, validation encodes one additional
teacher origin window. They run on validation/test, not training steps, and do not
change the training loss.

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

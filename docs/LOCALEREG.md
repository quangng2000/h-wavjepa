# LocaLeReg for H-WavJEPA

LocaLeReg is the selected regularizer in the upstream-backed training presets.
The online waveform encoder, context Transformer, upper MLPs, and predictors
remain trainable end to end. EMA targets retain their existing update rule.
The original masked objective and three future-prediction terms are retained.

## Evidence and implementation

The source is [LocaLeJEPA: Maximum Entropy Representations under Geometric
Priors](https://arxiv.org/abs/2610.28458), Section 4, with the authors'
[implementation](https://github.com/alessioborgi/LocaLeJEPA/blob/771361c18d37e65df3dc188761ffb26751ce4e37/src/losses.py)
pinned to `771361c18d37e65df3dc188761ffb26751ce4e37`. The vendored file is byte-exact,
with Git blob and SHA-256 hashes in its [manifest](../src/vendor/localereg/UPSTREAM.json)
and the original [MIT license](../third_party/licenses/LocaLeJEPA.txt).

The authors cluster detached embeddings with cosine k-means, subtract each
retained cluster's mean, and match the residual distribution to a uniform ball
of radius `sqrt(d + 2)` (identity covariance). Matching uses random normalized
projections and a characteristic-function discrepancy. Clusters with fewer
than four examples are skipped; if none survive, their code runs the centered
ball statistic on the entire batch. Retained cluster losses are equally weighted.

The paper's geometric assumptions and vision experiments motivate the method;
they do not establish that it improves this audio model. These presets are an
audio experiment, not a reproduction of the full LocaLeJEPA recipe.

## Audio adaptation

At level `l` and observed time `t`, the training population consists of **one
state from each example across all DDP ranks**. An autograd-aware all-gather
collects equal local batches; every rank evaluates the same loss and DDP's
parameter-gradient averaging gives the global-population gradient. The adapter
calls the authors' loss on shape `[1, global_batch, width]`, independently for
every time and level, then averages over time. It does not treat different times as image-style augmented views or merge
them into a larger batch. Clustering uses detached states; gradients flow through
the centered original states into all online layers that produced them.

```text
recordings at time t → states in level l → detached cosine k-means assignments
                                       → per-cluster mean subtraction
                                       → uniform-ball matching → R[l,t]

L = masked_MSE + sum_l prediction_weight[l] * future_MSE[l]
               + regularization_weight * sum_l mean_t R[l,t]
```

There are **three learned levels** and **up to four neighborhoods within each
level/time batch**. The neighborhood count is not another hierarchy. “Local”
describes proximity in embedding space, not a short audio duration. The mechanism
does not guarantee that neighborhoods correspond to speakers or sound classes.

| Setting | Audio default | Boundary |
| --- | --- | --- |
| `regularizer_kind` | `localereg` | `sigreg` selects the previous Gaussian control. |
| `localereg_clusters` | 4 | Experimental choice; upstream loss default is 8. |
| `localereg_min_cluster_size` | 4 | Small clusters skipped, with whole-batch fallback. |
| `localereg_kmeans_iters` | 10 | Uses upstream's torch backend. |
| `localereg_fixed_shape_kmeans` | false | Optional fixed-shape Lloyd updates; same cluster assignments and official BallReg loss. |
| `sigreg_projections` | 128 | Legacy field name shared by both regularizers; upstream ball-loss default is 256. |
| `localereg_n_quad` | 256 | Higher than upstream's 64 to resolve width 768. |
| Frequency knots/range | 17 / `[0,3]` | Upstream defaults. |
| `regularization_weight` | 0.1 | Initial experiment setting, not a tuned optimum. |
| AudioSet H100 population | 32/rank × 2 ranks = 64 | Exercised in three full-model generated-audio updates on two H100s; sustained real-data throughput unmeasured. |

The performance candidate presets enable `localereg_fixed_shape_kmeans`. Its
[attributed adapter](../src/objectives/localereg_fast.py) replaces dynamic boolean
indexing inside the detached k-means iterations with fixed-size scatter counts
and `torch.where`. The vendored reference remains unchanged; random initialization,
iteration count, cluster filtering, equal weighting, and the official BallReg
function are retained. Tests compare assignments, loss, gradients, diagnostics,
and RNG state against the reference, including empty clusters and global fallback.
This option records its implementation choice in checkpoint metadata. It is not
a measured speedup claim until a sustained GPU benchmark completes.

At width 768, 64 quadrature nodes give a maximum characteristic-function error
over 0.2 on the 17-point grid against an independent float64 512-node calculation.
The selected 256 nodes agree within `5e-6` in the numerical test. This is a check
at the configured width/grid, not an error guarantee for arbitrary dimensions.

The upstream statistic multiplies each cluster's discrepancy by its sample
count before averaging across clusters. Increasing the number of neighborhoods
therefore changes both geometry and loss scale; equal coefficients do not imply
equal regularization strength between SIGReg and LocaLeReg. Tune coefficients
and optimizer settings on validation data under comparable tuning budgets.

## Batches, diagnostics, and comparisons

One-device real-data presets request 32 examples per batch. The two-H100 AudioSet
preset gathers 32 per rank into a global population of 64 and drops incomplete
training batches to keep that population fixed. Changing the per-rank batch
changes the statistic and possibly the retained neighborhoods. **Gradient
accumulation does not enlarge the population used by a single regularizer call.**
The minimum two-example smoke configuration exercises the global fallback and
does not demonstrate local behavior.

BF16 states are gathered/cast with gradient tracking, then the statistic runs in
FP32 with autocast disabled. Random projections and cluster initialization use
synchronized, reproducible seeds. No additional world-size loss divisor is needed.
Two-process CPU tests compare losses and DDP gradients against a concatenated
single-process reference. A tiny two-H100 CUDA BF16 train/resume test passed.
The full model then completed three generated-audio updates at the configured
global population of 64. Peak allocated memory was approximately 29.4 GiB per GPU in that short
run; [validation evidence](VALIDATION.md) separates these checks from sustained
real-data throughput and downstream-quality claims.

Validation/test statistics use local populations without all-gather, and
multi-device held-out sampling can repeat tail entries. Training and validation
regularizer values therefore need not have the same scale or neighborhoods.
For final evaluation, use one device and keep the batch size/order fixed.
See the [AudioSet guide](AUDIOSET.md) for exact sampling and launch settings.

For each zero-indexed level, Lightning logs `localereg_configured_clusters`,
`localereg_effective_clusters`, `localereg_kept_fraction`, and `localereg_fallback`
under `train/level_0/`, `val/level_0/`, etc. Values are averaged over observed
times and, for epoch metrics, over batches. A fallback value of one means every
time in that batch used the global centered-ball statistic. `kept_fraction=0`
on fallback means no local cluster survived; the fallback still uses all examples.
The two-example default smoke run deliberately exercises this fallback.

Use `experiment=upstream_sigreg` for the random-initialization SIGReg control,
or `experiment=upstream_pretrained model.regularizer_kind=sigreg` for its
pretrained counterpart. Keep data, initialization, horizons, strides, and
training/evaluation budgets matched. Evaluate frozen low/middle/high readouts
with the existing HEAR route; lower pretraining loss alone is not a downstream win.

Checkpoints record the regularizer kind, settings, and source revision. An old
checkpoint without these fields retains SIGReg semantics. Resume rejects a
changed regularizer, cluster setting, quadrature setting, or source revision.
Select `model.regularizer_kind=sigreg` when resuming an older run. New presets
also select matched student/teacher normalized top-layer readouts; old checkpoints
retain their previous readout semantics. Runtime support includes CPU/CUDA FP32
and BF16 mixed precision with DDP. The short two-H100 hardware smoke passed;
full AudioSet training, sustained throughput, and quality remain unvalidated.

## What Lean establishes

[LocalRegularization.lean](https://github.com/quangng2000/hjepa-lean/blob/main/HJepa/LocalRegularization.lean)
proves the following in exact real arithmetic for a **fixed, nonempty cluster**:

```text
mean(z + a) = mean(z) + a
(z + a) - mean(z + a) = z - mean(z)
```

Consequently any fixed loss of centered residuals is unchanged, including
weighted sums over fixed clusters. This explains why centering does not force
every cluster centroid to the origin. It does not prove that recomputed cosine
k-means assignments are invariant, that floats implement exact arithmetic, or
that training preserves useful information or beats SIGReg on audio. Tests check
numerical behavior and gradient routing; downstream experiments remain necessary.

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

At level `l` and observed time `t`, the population consists of **one state from
each recording in the minibatch**. The adapter calls the authors' loss on shape
`[1, batch, width]`, independently for every time and level, then averages over
time. It does not treat different times as image-style augmented views or merge
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
| `sigreg_projections` | 128 | Legacy field name shared by both regularizers; upstream ball-loss default is 256. |
| `localereg_n_quad` | 256 | Higher than upstream's 64 to resolve width 768. |
| Frequency knots/range | 17 / `[0,3]` | Upstream defaults. |
| `regularization_weight` | 0.1 | Initial experiment setting, not a tuned optimum. |
| Real-data batch size | 32 | GPU memory and throughput unmeasured. |

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

Real-data presets request 32 recordings per batch. Reduce this only after
checking the retained neighborhoods. Gradient accumulation does not enlarge the
population used by a single regularizer call. The minimum accepted batch is two;
this permits smoke runs and small tails, but does not ensure local behavior.

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
Select `model.regularizer_kind=sigreg` when resuming an older run. CPU/CUDA FP32
single-device execution is supported by the adapter; only CPU has been tested.

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

# Learned Conv1D aggregation

`aggregation_mode: conv1d` replaces causal averaging with a learned, full
channel-mixing Conv1D **before** each upper-level MLP. It operates on completed
audio-window embeddings, not the backbone's finer acoustic tokens. The low
representation is retained, and extraction still returns all three levels.

This is an audio research variant. Neither the implementation checks nor the
Lean results establish that it improves spoofing detection or speech understanding.
Evaluate it against the matched mean-pooling baseline on held-out downstream data.

## Matched two-tap geometry

The presets retain cumulative strides `[1,2,4]`, kernels
`aggregation_windows: [1,2,2]`, five observed windows and horizons `[1,2,4]`.
Their nine-window training examples and original losses remain unchanged:

```text
medium[t] = G2(W2_previous low[t-1] + W2_current low[t])
high[t]   = G3(W3_previous medium[t-2] + W3_current medium[t])
```

Each weight is a feature-by-feature matrix. Both convolutions have no bias;
the existing MLPs retain their biases and nonlinearities. Initialization sets
each temporal weight to `I/2`, reproducing the mean-pooling arm's initial
features up to floating-point arithmetic without consuming additional random
numbers from the model-initialization stream. All Conv weights remain trainable.

Dense encoding uses dilation 1 at medium level and dilation 2 at high level,
then selects endpoints on the original stride grids. This implements stride-2
aggregation with its output phase aligned to those physical endpoints; a bare
`Conv1d(stride=2)` starting at array index zero need not have the same phase.
Dense states remain available for diagnostics and extraction.

There is **no boundary padding**: support widths remain `[1,2,4]`, and each
upper state is emitted only after its full support is available. This preserves
the existing forecast targets and avoids fabricating startup variation for the
temporal variance penalty. Student and EMA target paths use identical geometry.

At width 768, the two kernels add `2 × 768 × 768 × 2 = 2,359,296` online
parameters, plus a frozen EMA copy. GPU throughput and downstream quality have
not been measured; matching updates does not match compute or model capacity.

## Presets

- `upstream_conv1d_smoke`: generated-audio CPU integration preset.
- `audioset_conv1d`: matched to `audioset_temporal`.
- `audioset_conv1d_h100_four`: matched to `audioset_temporal_h100_four`.
- `audioset_nat_conv1d`: matched to `audioset_nat`.
- `audioset_nat_conv1d_h100_four`: matched to `audioset_nat_h100_four`.

Each changes only aggregation mode and descriptive tags relative to its control.
For example, run the small local check with:

```bash
uv run python -m src.train experiment=upstream_conv1d_smoke
```

Use the same data paths, seed, pretrained initialization and training budget as
the corresponding control. Nat variants still require the explicit staged index
and BRIR/noise manifest. These presets do not launch or modify a running trainer.

## Larger kernels and checkpoints

Three-tap kernels `[1,3,3]` have support widths `[1,3,7]`. They do not fit the
existing five-window context. One valid longer-context configuration uses
`context_windows=9`, `horizons=[1,4,8]`, and 17 total windows. These horizons keep
the target support entirely after the observed prefix. At the AudioSet preset's
1.01-second window size, that needs 17.17 seconds of real audio, so it is **not**
a drop-in setting for ten-second recordings. Padding or shortening audio windows
would constitute a different experimental recipe.

Aggregation mode and geometry are saved in checkpoint metadata. Checkpoints
without the new mode retain `mean` semantics; their weights load unchanged.
Exact resume rejects switching from mean to Conv1D or changing kernel geometry.
Starting the Conv arm from an existing model is a separately recorded weight
initialization operation, not an unchanged optimizer-state resume.

See [temporal aggregation](TEMPORAL_AGGREGATION.md) for state endpoints, target
supports, the variance penalty and limitations of the hierarchy experiment.

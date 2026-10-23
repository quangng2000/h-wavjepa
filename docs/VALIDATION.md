# Validation record

These checks establish software
behavior, not improved representation quality or reproduced benchmark scores.

## Nat integration, 23 October 2026

The later production-launch changes add resumable CPU asset staging and a
dependent Nat training bootstrap. The user explicitly waived the separate Nat
GPU probe. This is an operational choice, not new hardware-validation evidence;
launch receipts record `hardware_acceptance_passed=false`.

The final local CPU suite reports **1,005 passed, 4 skipped** in 135.21 seconds.
Two skips require BF16 CUDA DDP, one requires CUDA clustering kernels, and one
requires the preparation host's FFmpeg. Ruff lint and formatting (139 files),
`git diff --check`, and the frozen dependency lock check (114 packages) pass.
These are local results; the new commit's GitHub CI is recorded separately.

The [Nat implementation](WAVJEPA_NAT.md) has local tests for independent ear
encoders, channel-major shared masks, variable-length masked crops, joint ear
normalization, scene convolution and SNR, checksum-verified bounded asset
caching, pinned-source conversion and manifest assembly, Lightning updates,
checkpoint restore, and mono/stereo HEAR export. Tiny real training verifies
finite gradients and updates; silent PCM clips retain their distributed batch
positions with finite zero-energy rendering. The GPU probe's tests reject
missing BRIR/noise refills and frozen/shared/nonfinite ear encoders on any rank.

A separate checkout of the pinned WavJEPA source again matches small-model
tokens, predictions, teacher targets, masked loss and parameter gradients
exactly (maximum token difference zero). All 465 Nat checkpoint tensor shapes
and names match the published safetensors header, and fixed position encodings
match hashes of the actual released tensor bytes. Full Nat weights have not
been downloaded or loaded end to end in this validation.

The published room/noise assets have **not** been staged, and no Nat H100 test
or production training has run. The prior clean mono throughput result does
not validate Nat. Synthetic conversion/rendering tests and fake-S3 checkpoint
roundtrips establish code behavior only; real assets, streaming, capacity,
transfer timing and representation quality remain unverified.

## Earlier baseline validation, 22 October 2026

The earlier local suite reported **652 passed, 3 skipped** in 148.95 seconds.
The skips require two BF16-capable CUDA devices, CUDA for the optional clustering
parity check, and FFmpeg for the preparation integration test. The earlier
two-H100 smoke and CPU preparation-host checks are recorded separately below.
Ruff lint, Ruff formatting (123 files), and `uv lock --check` (114 packages) pass.
Previously recorded GitHub CI runs passed on Python 3.11, 3.12 and 3.13, plus
the separate official HEAR smoke; this local run is not a new CI result.

## Official WavJEPA reuse

- Source: `labhamlet/wavjepa` commit `6f4e1b1f18316c50768c8523f325ccffeaa6985b`.
- All original file hashes listed in `src/vendor/wavjepa/UPSTREAM.json` were
  checked against a separate original checkout before importing its model core.
- A small independent comparison matches contextual tokens, masked predictions,
  normalized teacher targets, masked loss and parameter gradients exactly.
- The actual official `labhamlet/wavjepa-base` weights at revision
  `6be4a5093f6c9adbbd55e00b6e5b8f067aa03345` were downloaded and loaded strictly.
  SHA-256: `988546976c453353c14ebb0798f275ea5eb95270d75eed506ea455c5b49f6be0`.
- The released model produced `[2, 200, 768]` contextual tokens with maximum
  absolute difference **0.0** against the separate original source. Predictions,
  teacher targets and masked loss also matched exactly. This compares identical
  normalized input windows and fixed masks, not the original full data pipeline.
- The full default model constructs with exactly 32,160 samples. Tests also
  cover strict import failures and offline loading without remote Python code.
- The AudioSet preset uses 16,160-sample windows. Tests verify strict learned-weight
  loading while regenerating the fixed sinusoidal positions for 100 tokens; the
  released checkpoint's original 200-token positions must first match the expected
  formula. This shorter-window adaptation is not an upstream training reproduction.
- The active positional formula is now independently implemented. Independent
  source comparison still gives exact tokens, predictions, targets, loss and
  parameter gradients; checkpointed execution also preserves attention masks.

Reproduce with `python -m src.verify_upstream --upstream /path/to/wavjepa`;
add `--pretrained` after explicitly running `python -m src.pretrained`.
CI runs the small source comparison against the pinned original repository.

## Training and inference

The CPU suite covers gradients through all three spaces, future-input invariance,
window/horizon indices, independent SIGReg value/gradient parity, EMA timing,
warmup/cosine schedules, accumulation, checkpoint resume and legacy compatibility.
It also exercises actual WAV training/evaluation, explicit dynamics embedding
export, checkpoint hashes and finite-value/input validation.

LocaLeReg checks cover byte-exact upstream source hashes, timewise adapter
argument/reduction/gradient plumbing, fixed-cluster translation invariance,
cluster filtering and fallback reporting, and independent float64
target-distribution quadrature at width 768. The timewise adapter comparison calls
the same pinned upstream loss on both sides; it does not independently validate
that loss. The analytic target-distribution checks and fixed-shape implementation
comparison below provide separate numerical evidence. Tiny CPU training checks
cover local clusters, gradients into the online waveform/context encoders and
upper MLPs, checkpoint inference, and resume.
These tests do not measure learned audio quality. The companion Lean build and
axiom audit check only the stated [centering property](LOCALEREG.md#what-lean-establishes).

Two-process Gloo/DDP tests compare distributed SIGReg and LocaLeReg values and
parameter gradients against one global-batch reference. Separate externally
launched `torchrun` tests exercise FP32 and BF16 mixed precision, two-layer/top-two
normalized readouts, gradient checkpointing, accumulation, real 10-second WAVs,
rank-disjoint sampling and checkpoint resume. Both ranks finish with identical
online/teacher state hashes, and EMA updates equal optimizer steps. The integration
uses a small model and short windows; it is not an H100 throughput or memory test.

A separate two-rank **FP32 full-objective** test compares loss, every trainable
parameter gradient, two SGD-with-momentum updates, and online/EMA state against
an independently assembled single-global-batch reference. It covers SIGReg and
LocaLeReg, hierarchical and flat controls, a two-layer/top-two readout, and DDP
with `find_unused_parameters=false`. Masks and random sketches are controlled
and dropout is zero so the objectives are actually identical. This catches
world-size loss-scaling errors; it does not assert exact equality for arbitrary
BF16 data streams. Injected one-rank invalid inputs, losses and gradients also
verify collective rejection before optimizer/EMA changes.

HEAR regression tests use multiple layers and a nonidentity final LayerNorm to
distinguish the trained normalized top-k readout from the historical readout.
Export manifests record that choice and identify duplicate flat levels. EMA
timing tests now compare parameter values with the post-optimizer student, so
an update performed before the optimizer would fail. Synthetic temporal-collapse
fixtures validate the [held-out diagnostics](TEMPORAL_DIAGNOSTICS.md); these
diagnostics have not yet established useful temporal structure in a trained model.

Data tests separately load a literal 160,000-sample recording into nine complete
16,160-sample AudioSet windows without padding, verify segment bounds and epoch
crop changes with persistent workers, and reject recording leakage across splits.
Export tests preserve manifest offsets for both current and legacy checkpoints.
Older checkpoints keep their historical readout definition; new training uses
matching normalized top-k readouts for the student and EMA upper maps.

Run:

```bash
uv sync --frozen --extra hear --extra data --extra nat
uv run --extra hear --extra data --extra nat ruff check .
uv run --extra hear --extra data --extra nat ruff format --check .
uv run --extra hear --extra data --extra nat pytest
uv run python -m src.train experiment=upstream_smoke
```

## Execution and streaming regressions

Optional masked-token compaction is compared against the dense path for retained
predictions, masked loss, teacher targets and every parameter gradient. Cases
include uneven masks, one visible token, unmasked rows, checkpointed execution,
CPU BF16, zero-dropout enforcement and checkpoint compatibility. These numerical
checks do not measure a GPU speedup.

The fixed-shape LocaLeReg adapter preserves reference cluster assignments, loss,
gradients, diagnostic values and RNG state in CPU comparisons, including empty
clusters and whole-population fallback. Two-process Gloo tests also compare its
distributed gradients with the official global-population reference. Finite
gradient checks still reject NaN/Inf before optimizer or EMA changes, while
accepting large finite gradients whose norm would overflow. Training with sampled
diagnostics produces identical model state to logging every update in the tested
CPU run. The fast presets preserve global batch 64 and the full update schedule.

Actual Lightning runs over infinite iterators verify periodic latest checkpoints
and a final checkpoint when validation continually worsens. Retention is bounded
to one latest regular file, a `last.ckpt` symlink and one optional best file.
An official Lightning Timer stops the iterator with `min_epochs=0` and preserves
the final optimizer state. Tests also construct the fast presets' callbacks and
exercise ordinary AdamW with BF16 mixed precision and gradient clipping; the
unsupported fused-optimizer combination is rejected during configuration checks.
Fault injection checks incomplete serialization and failed alias publication:
the preceding `last.ckpt` remains readable. Resuming the first periodic checkpoint
in the same directory preserves that stable alias. Successful saves bound
retention; hard kills can leave orphan files, as detailed in the streaming guide.

The sharded-input tests exercise actual WebDataset TAR decoding, source and
manifest checksum rejection, private-read fixtures, bounded caches, real worker
processes, rank-disjoint training and two-process collective validation. Resume
tests reject an index change or unacknowledged worker-sequence restart. They do
not establish exact iterator-position recovery, which is not implemented.
Additional checks cover advancing and recording a restarted stream's seed,
whole-object download deadlines across SDK retries, and temporary benchmark
audio/checkpoints staying outside the platform's `/output` artifact directory.

CPU tests for the bounded GPU benchmark programs verify local/S3 pilot checksums,
safe extraction, unchanged global-batch/loss/schedule settings, all-rank report
agreement, validation evidence and full checkpoint metadata. They do not turn a
configured GPU probe into an executed throughput result. The separate
[S3 read probe](validation/s3-read-probe-2026-10-22.json) measured one verified
625,715,200-byte read plus SHA256 at 96.1 MB/s on a CPU node; it excludes disk
caching, audio decoding and concurrent GPU training.

## Real AudioSet CPU pilot

The CPU-only preparation pilot converted 2,128 clips from the pinned WavJEPA
AudioSet source: one complete training shard and 64 attempts each from one
validation and test shard. It published 1,953 training, 63 validation and 64 test
clips; 48 clips were rejected. All three source archives were SHA256-verified,
all accepted WAV checksums were checked, and the experiment-aware two-H100
manifest preflight passed with no padding and disjoint recording IDs.

All 28 preparation tests passed on the CPU host, including real FFmpeg conversion.
The 771,368,960-byte pilot archive was uploaded to private object storage and
downloaded back with the same SHA256. This is a small validated dataset, not
full-corpus completion or a GPU throughput measurement. The
[machine-readable evidence](validation/audioset-cpu-pilot-2026-10-22.json)
records the source hashes, counts, runtime and limitations. See the
[preparation guide](PREPARE_AUDIOSET.md) for reproducible commands.

The rolling S3 stager passed all 47 preparation and staging tests on the CPU
host. Its real pilot uploaded one 625,715,200-byte training shard and verified
the full object SHA256 before removing temporary WAVs. A second invocation
preserved the verified index without re-uploading the source. The
[staging evidence](validation/audioset-staging-pilot-2026-10-22.json) records
these checks; its index remains explicitly partial at one of 889 sources.
See the [staging guide](STAGE_AUDIOSET.md) for storage and resume requirements.

## Two-H100 smoke

On 22 October 2026, commit `4a9b09a80b4e1fac2bb63a856606574a96e78e8f`
completed a bounded cloud test on **two NVIDIA H100 80GB HBM3 GPUs**, using
PyTorch 2.13.0+cu126, CUDA 12.6 and Lightning 2.6.6. The CUDA/BF16 variant of
`tests/test_dynamics_ddp.py` passed with no skips, covering two-rank training,
gradient checkpointing, accumulation, WAV sampling, checkpoint resume, equal
rank-state hashes, and EMA timing.

The same job then completed **three full-base optimizer updates**, with 32
generated recordings per GPU (64 globally), nine 16,160-sample windows per
recording, BF16 mixed precision, checkpointing, the pointwise hierarchy, and
LocaLeReg. Both ranks finished with three EMA updates. Peak CUDA allocation was
**29.405 GiB per GPU**, with **33.344 GiB reserved**. The slower rank's three
observed batch times were 1.659, 0.905 and 0.663 seconds.

These are preliminary capacity and correctness observations. Three updates on
generated audio do not measure corpus loading, checkpoint/validation overhead,
stable throughput, convergence, or full-training duration. The probe uses a
short learning-rate/EMA schedule and downloads neither AudioSet nor pretrained
weights. A longer pilot on the intended corpus is still required.

The [machine-readable record](validation/h100-smoke-2026-10-22.json) contains
both ranks' measurements. Reproduce with the
[bounded smoke procedure](../scripts/GPU_SMOKE.md); the image verifies its
Torch/CUDA versions before GPU allocation.

## HEAR

The adapter tests cover low/medium/high and fine backbone readouts, variable clip
lengths, bounded extraction, timestamps and export of scene/event task metadata.
The separate runner delegates interval labeling, packaging, probe selection and
scoring to a pinned official HEAR evaluation kit. See [HEAR](HEAR.md) for the
environment and reproducible synthetic end-to-end smoke commands.

Both official scene classification and event-onset probe runs completed locally
with finite scores and their `prediction-done.json` markers. The event fixture
selected epoch 0 while training finished later, exercising the Lightning 1.9
compatibility fix for chosen-epoch postprocessing. The other guarded patch makes
TensorFlow optional for PyTorch-only packaging. These generated-data scores are
integration evidence only; they are not reported as model-quality results.

## Temporal aggregation and frozen-readout checks

The optional [temporal model](TEMPORAL_AGGREGATION.md) has CPU coverage for valid
causal support, target-window alignment, future-audio isolation, within-recording
variance pressure and old pointwise checkpoint compatibility. The generated-audio
Hydra preset completed one epoch with validation. Two-process CPU FP32 and BF16
tests completed training and checkpoint resume; a separate DDP test compares
gradients with a single global batch, including an independently computed temporal
penalty. HEAR tests compare chunked extraction with full-history encoding and
cover support-center timestamps, partial tails and short-clip padding.

The complete local suite passed 768 tests with three environment-dependent skips
before the frozen-readout probe and extra temporal BF16 variants were added. The
additional run passed 28 probe/DDP checks, including 24 new probe tests and the
new temporal CPU BF16 train/resume test; its two CUDA variants were skipped on
this CPU host. The probe tests cover whole-cohort moment merging, batch-partition
invariance, recording deduplication, official-weight verification and fixed crop
provenance. These test fixtures do not measure the released model on AudioSet.

No real benchmark or model-quality result is claimed. The H100 evidence above
applies to the pointwise model. Temporal aggregation has not been validated on
H100 hardware, and its throughput is unmeasured. The abstractions, forecasting
heads and temporal floor still require controlled training and downstream
comparisons; the pointwise presets remain available unchanged.

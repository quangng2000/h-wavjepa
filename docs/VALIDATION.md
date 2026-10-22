# Validation record

Validated locally on CPU on 22 October 2026. These checks establish software
behavior, not improved representation quality or reproduced benchmark scores.

The local suite reports **454 passed, 1 skipped**. The skipped case needs two
BF16-capable CUDA devices and subsequently **passed on two H100s**, as recorded
below. Ruff and an independent original-source comparison also pass. GitHub CI
passed on Python 3.11, 3.12 and 3.13, plus the separate official HEAR smoke.

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
value/gradient agreement, fixed-cluster translation invariance, cluster filtering
and fallback reporting, and independent float64 target-distribution quadrature at
width 768. Tiny CPU training checks cover local clusters, gradients into the
online waveform/context encoders and upper MLPs, checkpoint inference, and resume.
These tests do not measure learned audio quality. The companion Lean build and
axiom audit check only the stated [centering property](LOCALEREG.md#what-lean-establishes).

Two-process Gloo/DDP tests compare distributed SIGReg and LocaLeReg values and
parameter gradients against one global-batch reference. Separate externally
launched `torchrun` tests exercise FP32 and BF16 mixed precision, two-layer/top-two
normalized readouts, gradient checkpointing, accumulation, real 10-second WAVs,
rank-disjoint sampling and checkpoint resume. Both ranks finish with identical
online/teacher state hashes, and EMA updates equal optimizer steps. The integration
uses a small model and short windows; it is not an H100 throughput or memory test.

Data tests separately load a literal 160,000-sample recording into nine complete
16,160-sample AudioSet windows without padding, verify segment bounds and epoch
crop changes with persistent workers, and reject recording leakage across splits.
Export tests preserve manifest offsets for both current and legacy checkpoints.
Older checkpoints keep their historical readout definition; new training uses
matching normalized top-k readouts for the student and EMA upper maps.

Run:

```bash
uv sync --frozen --extra hear
uv run --extra hear ruff check .
uv run --extra hear ruff format --check .
uv run --extra hear pytest
uv run python -m src.train experiment=upstream_smoke
```

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

No real benchmark or model-quality result is claimed. Distributed and
mixed-precision evidence comes from generated audio on CPU and the short H100
smoke above. The new abstractions and forecasting heads still require training
and controlled downstream comparisons. The hierarchy remains the pointwise
baseline: strides change the sampled state grid, and the upper MLPs do not pool
adjacent windows.

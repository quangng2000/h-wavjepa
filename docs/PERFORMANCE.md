# H100 training performance

Keep the experiment fixed when comparing speed: global batch **64**, nine
1.01-second windows per recording, the full WavJEPA base, pointwise hierarchy,
LocaLeReg, gradient clipping, the **375,000-update** LR schedule, 100,000-update
LR warmup, and the existing 100,000-update EMA ramp.
These changes reduce execution overhead; they do not establish representation
quality or turn the pointwise baseline into temporal aggregation.

## Measured local-audio throughput

On H100 80 GB GPUs, PyTorch 2.13.0 / CUDA 12.6 / Lightning 2.6.6:

| Configuration | Updates/s | Peak allocated memory/GPU | 375k training hours | Training GPU cost |
| --- | ---: | ---: | ---: | ---: |
| Two GPUs, dense with checkpointing | 1.467 | 29.4 GiB | 71.0 | $537 |
| Two GPUs, compact without checkpointing | 1.944 | 42.3 GiB | 53.6 | $405 |
| Four GPUs, compact without checkpointing | 2.747 | 23.0 GiB | 37.9 | $573 |

Each result measures 200 complete optimizer updates after 25 warmup updates on
the verified AudioSet pilot. All ranks completed 225 optimizer and EMA updates,
with changed online weights and full checkpoints. The two-GPU compact path was
32.5% faster than this dense/checkpointed control, which already includes the
other runtime optimizations. [Raw results and receipt](validation/h100-performance-2026-10-22.json).

Costs use the observed standard batch rate of $0.063/GPU-minute. They exclude
remote input, validation, checkpoint upload and interruptions. Adding an assumed
15% runtime margin gives roughly **61.6 hours / $466** on two GPUs or
**43.6 hours / $659** on four. This margin is a planning assumption, not measured
full-run overhead. Calendar discounts are not included.

The initial live streaming probe was stopped during startup. Its input workers
used the unsafe Linux fork-after-NCCL combination. Both loaders now explicitly
spawn worker processes; corrected live streaming validation is required before
these compute projections qualify a full training run.

## Available controls

`experiment=audioset_h100_fast` uses two GPUs and 32 recordings per rank;
`experiment=audioset_h100_four` uses four GPUs and 16 recordings per rank.
Both gather the same 64-recording population for regularization and select the
measured compact path without gradient checkpointing, with two workers per rank.
The streaming probe uses one worker per rank and measures input separately.

- Finite-gradient checks reduce their flags on the GPU before one host check per
  device. Nonfinite gradients still abort the optimizer and EMA update.
- Training diagnostics are sampled every 50 updates. Every update still computes
  the full objective; every validation batch still contributes its metrics.
- Optional fixed-shape LocaLeReg clustering avoids dynamic indexing inside Lloyd
  iterations. The original BallReg loss, seeded clustering, cluster filtering,
  projection count and quadrature stay the same. The vendored reference remains
  byte-for-byte unchanged; CPU and CUDA value/gradient comparisons cover the adapter.
- `model.backbone.config.compact_masked_tokens=true` packs visible context and
  target-mask positions before the official Transformers, then restores original
  positions. The dense teacher is unchanged. This requires zero dropout and is
  mathematically equivalent on retained positions, with normal floating-point
  reduction differences. The original presets retain the dense path.
- `model.backbone.config.use_gradient_checkpointing=false` removes activation
  recomputation. The measured fast presets use this setting; the original
  two-H100 preset retains checkpointing for lower memory use.
- Persistent input workers prefetch batches. For stored shards, follow the
  [WebDataset input guide](SHARDED_AUDIO.md) and benchmark cold cache refills.

The fast presets retain ordinary AdamW: Lightning 2.6.6 rejects its fused variant
with mixed precision and nonzero gradient clipping. A GPU probe exposed this, and
configuration validation now rejects that combination before allocation.

## Measure before estimating a full run

`scripts/benchmark_gpu.py` compares dense/checkpointed, dense/uncheckpointed and
compacted paths on two and four H100s. It verifies a small real AudioSet pilot,
runs a CUDA regularizer parity test, then measures 200 updates after 25 warmup
updates per configuration. Timings include local WAV reads and new epoch crops.
It records all ranks, memory, optimizer/EMA counts, changed encoder weights and a
full checkpoint's write time. `Dockerfile.gpu-benchmark` pins the tested CUDA image
and uses the lockfile's Python dependency versions.

For the private pilot already staged by the preparation pipeline,
`Dockerfile.gpu-benchmark-local` verifies and embeds `benchmark-pilot.tar` during
the CPU build. Keep that archive out of Git. `HWAVJEPA_LOCAL_PILOT` selects it at
runtime; otherwise the script uses a read-only S3 key and bounds the download
to 120 seconds. SFCompute upload contexts must be **zstd-compressed TARs**.

```bash
python -u scripts/benchmark_gpu.py --output /output/benchmark \
  --variants compact_4 compact_2 dense_checkpointed_2 --steps 225 --warmup 25
```

This supervisor needs four visible GPUs and launches isolated two/four-rank
runs. Two-rank measurements still rent all four GPUs in this comparison job;
their projected production cost assumes renting only two.

The pilot is cached locally. It excludes full-corpus cache misses, validation,
checkpoint upload, queueing and preemption. The CPU S3 [read probe](validation/s3-read-probe-2026-10-22.json)
measured 96.1 MB/s for one checksum-verified 626 MB object while staging ran;
that result excludes disk caching, decode and concurrent GPU training.

For budgeting, compute `375000 / updates_per_second / 3600` training hours,
then add measured input/validation/checkpoint overhead and a margin. Multiply
the complete runtime by the provider's current rate and allocated GPU count.
Do not infer four-GPU throughput by doubling a two-GPU measurement, or double
the global batch while comparing a fixed optimizer-update budget.

After local compute timing, `scripts/benchmark_streaming_gpu.py` exercises the
production S3 loader on a frozen partial index with a fresh cache, one worker per
rank, and global batch 64. It requires cache refills after warmup on every rank,
times held-out validation separately, and saves/reloads a full checkpoint to
verify its model, optimizer, scheduler and input identity. Supply the read-only
credentials and index environment variables described in the streaming guide.

```bash
torchrun --standalone --nproc_per_node=4 scripts/benchmark_streaming_gpu.py \
  --devices 4 --compact --steps 400 --warmup 25 --workers 1 \
  --output /output/streaming-benchmark
```

These scripts do not provision resources or authorize spending. An external job
must impose a runtime/cost bound. A passing local-pilot benchmark alone does not
establish full-corpus end-to-end throughput.

## Long runs

Use `callbacks=step_checkpoint trainer.min_epochs=0` with the infinite streaming
iterator. This preserves a latest full checkpoint every 5,000 updates and at a
normal final stop, independently of validation improvement. Lightning's official
`trainer.max_time` can then stop cleanly. Files are local until exported; see
[checkpoint durability and resume](SHARDED_AUDIO.md).

SFCompute's checked batch-job duration ceiling was 720 minutes, with a self-serve
increase available up to 1,440 minutes. A longer run needs planned checkpointed
segments and allowance for queue delays. Merely setting the platform's
`resume=checkpoint` does not wire a training program to discover, load or capture
its checkpoint. The current streaming loader explicitly restarts worker sequences
on an opted-in resume; it does not restore prefetched sample positions.

# AudioSet training

The AudioSet presets keep the pointwise hierarchy and fit a chronological
excerpt inside a real ten-second clip. The two-H100 preset enables DDP, BF16
mixed precision, and activation checkpointing. CPU distributed tests exercise
these integration paths; **target-GPU memory, throughput, and wall time remain
unmeasured**. No GPU result should be inferred from the preset name.

## Experimental recipe

| Setting | AudioSet adaptation |
| --- | --- |
| Input | Mono PCM16 WAV, 16 kHz |
| Window | 16,160 samples = 1.01 s, 100 backbone tokens |
| Excerpt | Nine consecutive windows = 145,440 samples = 9.09 s |
| Observed history | First five windows = 5.05 s |
| Future target indices | 5, 6, 8, counted from zero |
| Target-end horizons after observed history | 1.01 s, 2.02 s, 4.04 s |
| State strides | Low/medium/high: 1, 2, 4 |
| Training budget | 375,000 optimizer steps |
| Learning rate | Peak 0.0002; 100,000-step warmup, then cosine decay |
| EMA | 0.999 to 0.99999 over the first 100,000 optimizer steps |
| Two-H100 batch | 32 examples per rank, 64 globally; incomplete training batches dropped |

Every target represents **one future 1.01-second window**. G2/G3 transform one
lower state each; their stride does not pool the intervening audio. Student
and EMA upper maps now receive the same normalized top-eight-layer readout
transformation. The masked branch's existing teacher tokens are reused for
future targets, avoiding a duplicate nine-window teacher pass.

The original [WavJEPA recipe](https://arxiv.org/html/2509.23238v1#S3.SS2) reports
1.74 million AudioSet clips, eight random two-second crops per clip, and 375,000
steps on two H100s. Matching the step count here does **not** match its crop
sampling, optimizer batch, audio exposure, or compute. Record the available
AudioSet subset and manifest hashes: missing source clips can change the cohort.
No elapsed training time is inferred from that paper.

`upstream_hierarchical` and `upstream_pretrained` retain nine 32,160-sample
windows: 18.09 seconds total, 10.05 seconds observed, and target-end horizons
2.01/4.02/8.04 seconds. They require longer recordings and reject ten-second clips.

## Data and manifests

Prepare/download the audio separately before launching GPUs. The loader does not
download, resample, concatenate clips, or create splits. Paths are relative to
the manifest's directory unless absolute:

```json
{"path": "audio/clip.wav", "recording_id": "original-recording-id"}
```

Optional fields are `group_id`, `recording_id`, `offset_samples`, and
`segment_samples`; unknown fields fail validation. Offsets and lengths are
integers in **samples at 16 kHz**, not seconds:

```json
{"path": "audio/long.wav", "offset_samples": 0, "segment_samples": 160000, "recording_id": "long-recording"}
{"path": "audio/long.wav", "offset_samples": 160000, "segment_samples": 160000, "recording_id": "long-recording"}
```

`offset_samples` defaults to zero. Omitting `segment_samples` selects the
remainder of the file. A segment must fit inside the recording and contain at
least 145,440 real samples for the AudioSet presets. Padding is forbidden.
Multiple distinct segments of one recording may occur in the **same** split;
physical recordings and supplied recording/group IDs must not cross splits.
If an ID field appears anywhere, it must be supplied on every row. These checks
cannot identify copied audio in different files, so preserve source IDs.

Each training epoch chooses a deterministic random chronological crop inside
each segment using the configured seed, epoch, and manifest index. The crop
is independent of worker scheduling and rank assignment, including persistent
workers. Validation, test, and embedding export always start at the declared
segment offset. Keep manifests and audio immutable during training.

Run preflight with the experiment you will train:

```bash
uv run python -m src.validate_data --experiment audioset_h100 \
  --train /data/audioset/train.jsonl \
  --val /data/audioset/val.jsonl \
  --test /data/audioset/test.jsonl
```

It checks format, bounded payload extents, segments, split separation, required
length, and per-rank batch minimums. For the default two-GPU configuration,
training needs at least 64 entries and validation/test each need at least four.
Options such as `--batch-size`, `--minimum-batch-size`, `--world-size`,
`--num-samples`, `--require-full-length`, and `--drop-last-train` can mirror
explicit training overrides. Without `--experiment`, legacy preflight defaults
allow padding; use the experiment-aware command for forecasting.

## Two H100s

Install into the GPU environment and choose a **new directory on persistent
storage** for each invocation:

```bash
uv sync --frozen --extra hear
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.train \
  experiment=audioset_h100 \
  data.train_manifest=/data/audioset/train.jsonl \
  data.val_manifest=/data/audioset/val.jsonl \
  data.test_manifest=/data/audioset/test.jsonl \
  hydra.run.dir=/checkpoints/hwavjepa/run-001
```

This command starts random initialization. Add `model.backbone.pretrained=true`
to initialize from the official WavJEPA base release. Downloading is explicit,
pinned, checksum-verified, and executes no Hub Python code. The shorter window
uses 100 positional tokens instead of 200. The loader verifies the original
fixed sinusoidal tensors before regenerating only those two nonlearned position
arrays; all learned tensors retain strict key, shape, dtype, and finite checks.
This positional adaptation is not a claim that the shorter-window model matches
the original model's outputs.

For offline initialization, download with `uv run python -m src.pretrained`
ahead of time and set
`model.backbone.pretrained=false model.backbone.pretrained_path=/path/model.safetensors`.
Use `experiment=audioset_pretrained` for the one-device FP32 equivalent.

The WAV datamodule owns distributed sharding, so
`trainer.use_distributed_sampler=false` must remain set. Training drops the
incomplete distributed tail, preserving equal full local batches without
repeating examples. Each new epoch reshuffles the omitted tail. Regularizers
use differentiable all-gather to see 32 examples from each rank at the same
observed time: **64 total**, with FP32 statistics inside mixed-precision training.
`trainer.accumulate_grad_batches` increases the optimizer's effective batch;
it does not increase a regularizer invocation's population. If GPU memory
requires a smaller per-rank batch, report and validate that changed population.

Validation runs periodically during training. Regularizer validation statistics
are local to each rank, so their population differs from training's global
population. Distributed held-out sampling can pad the tail; it is suitable for
training diagnostics, but final metrics should use one device as below.

Benchmark real loading and training before the full budget. Keep the planned
learning-rate/EMA schedule fixed when stopping a pilot early. Hardware
validation and a reliable two-H100 duration estimate are still pending.

## Resume the planned budget

A full Lightning checkpoint restores weights, optimizer, EMA, and the resolved
schedule. It bypasses pretrained initialization downloads. Resume into a new
run directory, keeping model, regularizer, data, device layout, and schedule
settings consistent:

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 -m src.train \
  experiment=audioset_h100 \
  data.train_manifest=/data/audioset/train.jsonl \
  data.val_manifest=/data/audioset/val.jsonl \
  data.test_manifest=/data/audioset/test.jsonl \
  ckpt_path=/checkpoints/hwavjepa/run-001/checkpoints/last.ckpt \
  trainer.max_steps=375000 \
  hydra.run.dir=/checkpoints/hwavjepa/run-002
```

`trainer.max_steps` is the **total optimizer-step target**, not additional steps
after resuming. The AudioSet preset fixes `model.lr_total_steps=375000`,
`model.lr_warmup_steps=100000`, and `model.ema_anneal_end_step=100000`.
A partial run may stop at a smaller trainer maximum while retaining this planned
schedule; restore the full maximum on resume. Extending a completed cosine
schedule is rejected. Changing batch size, world size, or manifest order changes
the sample trajectory even if the optimizer-step schedule is preserved.

## Evaluation

Use one device to visit every held-out entry once. Keep the checkpoint's model
and data settings; these overrides change execution only:

```bash
uv run python -m src.eval experiment=audioset_h100 \
  trainer.devices=1 trainer.strategy=auto \
  ckpt_path=/checkpoints/hwavjepa/run-001/checkpoints/last.ckpt \
  data.train_manifest=/data/audioset/train.jsonl \
  data.val_manifest=/data/audioset/val.jsonl \
  data.test_manifest=/data/audioset/test.jsonl \
  hydra.run.dir=/checkpoints/hwavjepa/eval-001

uv run python -m src.embed --model-family dynamics \
  --checkpoint /checkpoints/hwavjepa/run-001/checkpoints/last.ckpt \
  --manifest /data/audioset/test.jsonl --output /new/embeddings \
  --num-samples 145440
```

The first command computes objective diagnostics on the held-out split. Changing
batch layouts changes regularizer statistics, so compare like-for-like settings.
Embedding export honors manifest segments and uses fixed crops. For downstream
quality, evaluate frozen low, medium, high, and backbone features through the
[HEAR interface and official runner](HEAR.md). Own-space prediction losses alone
do not establish useful semantics or improvement over the flat control.

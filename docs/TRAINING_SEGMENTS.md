# Bounded training segments

This runner supports clean mono pointwise and temporal models, plus the
binaural [Nat temporal model](WAVJEPA_NAT.md). Nat requires a completed, verified
training-asset manifest with a pinned SHA-256. The clean mono acceptance receipt
cannot stand in for a Nat hardware result.

`scripts/train_segment.py` runs one four-H100 segment, saves full Lightning state
to private S3, and exits. It does not rent GPUs, restart jobs, transfer credits, or
top up an account. The intended plan is to use at most the approved **$500 shared
credit allocation**, pause with a verified checkpoint before that allocation is
exhausted, and resume beyond that allocation after manual approval and top-up.
Planned segments within the remaining approved allocation do not need fresh
approval; each still needs a balance and spending-limit check before launch.

By default, training requires the completed AudioSet index and a successful
four-GPU streaming probe using **one spawned data worker per rank**, the same
model variant, and the exact completed index SHA256. The failed
fork-worker probe and the successful local-audio timing run do not pass this gate.
See [measured performance and remaining limitations](PERFORMANCE.md).

An explicit user decision can waive the separate hardware probe with
`--skip-streaming-acceptance --streaming-acceptance-waiver-reason '<reason>'`.
The run records `waived_by_user` and `hardware_acceptance_passed=false`, never a
passing test. This does not waive dataset checks, runtime validation, budget
limits, or checkpoint verification. Nat throughput and total cost remain
unmeasured until actual production results exist.

## Budget and launch bounds

Preview a ten-hour segment without credentials or network access:

```bash
python scripts/train_segment.py --plan-only \
  --segment-hours 10 --budget-usd 160 --maximum-rate-usd-per-hour 15.12
```

At this example rate, the printed platform cap is **605 minutes / $152.46** for
all four GPUs, including a five-minute backstop. Set the actual platform job to
the printed `max_duration_minutes`, **`max_restarts=0`**, and **`resume=none`**.
The rate is an input, not a live quote; check the provider's current total node
rate before each launch. The runner cannot inspect billing or enforce the shared
account balance using an S3 key.

Before **every** segment, the orchestrator must check remaining organization
credit, remaining approved allocation, credit expiry, and any concurrent charges.
Choose a segment ceiling that fits all of those limits, with separate allowance
for image builds and recurring storage. This runner's compute ceiling excludes
build/storage charges. Do not start three ten-hour segments merely because their
nominal compute total is below $500. Shorten the last segment as needed and pause
for manual top-up; there is no automatic personal-account fallback.

The local watchdog covers setup, resume download/deserialization, training,
validation, checkpoint publication, and process teardown. It stops the whole
owned process tree, including torchrun's detached rank sessions, before the
requested segment duration; the platform cap remains the final backstop.
Segments may be at most 11.5 hours. Defaults reserve the last
**30 minutes** for finalization and limit each checkpoint publication to **10
minutes**. These margins are conservative inputs, not measured full-corpus
validation or checkpoint upload durations. Measure the upload plus full readback
and validation tail before treating them as sufficient for production.

Segment runs raise the DDP collective timeout to at least the transfer budget
plus **120 seconds**: **720 seconds** with the default 600-second transfer budget.
This lets peers wait for rank zero's publication result; a longer preset timeout
is preserved. Ordinary H100 runs retain their 180-second timeout. The durable
callback rejects a shorter timeout at setup, including outside the segment runner.
The transfer deadline is checked cooperatively around SDK operations; the extra
time allows an in-flight request and the result broadcast to finish. It is not
an exact wall-clock guarantee; the segment watchdog still bounds the whole job.

## Required inputs and credentials

- A completed staging index, stored locally with its exact SHA256. Partial
  indexes are rejected, and every resumed segment must use identical index
  bytes. The pipeline never substitutes an updated index during resume.
- `streaming-benchmark-success.json` from a passing four-GPU streaming probe with
  one spawned worker per rank, cache refills on every rank, finite parameter
  updates, full held-out validation (apart from the global-batch tail), and a
  verified full checkpoint upload/readback/download roundtrip through private S3.
  Every rank and checkpoint must carry the selected model's geometry and the
  completed index hash. Temporal runs also require finite upper temporal-loss
  evidence; a pointwise report cannot authorize a temporal run.
- An existing private checkpoint bucket and an empty, dedicated run prefix,
  such as `training/training-run-20261023`. The last prefix component must equal
  `--run-id`. Only one writer may use this prefix at a time.
- Runtime environment credentials injected by the job/secret manager:
  standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` remain the **read-only
  AudioSet credentials**. Separately inject
  `HWAVJEPA_CHECKPOINT_ACCESS_KEY_ID` / `HWAVJEPA_CHECKPOINT_SECRET_ACCESS_KEY`
  for the checkpoint bucket, with GET/HEAD/LIST/PUT/DELETE access. Optional
  `HWAVJEPA_CHECKPOINT_SESSION_TOKEN` and `HWAVJEPA_CHECKPOINT_REGION` apply only
  to the checkpoint client. Missing separate credentials cause failure; the
  runner never replaces or falls back to the AudioSet credentials.

No bucket or key is created by the script. Keep keys out of command arguments,
Git, logs, images, and receipt files. The launch below uses placeholders only.

## One segment

Run the supervisor once, rather than wrapping it in `torchrun`; it launches four
ranks itself. Both output and working directories must be new for each segment.
Put the working directory outside the platform's automatically collected output
directory so cached shards and large local checkpoints are not exported twice.

```bash
python -u scripts/train_segment.py --temporal \
  --segment-hours 10 --budget-usd 160 --maximum-rate-usd-per-hour 15.12 \
  --finalization-reserve-seconds 1800 --transfer-timeout-seconds 600 \
  --run-id training-run-20261023 --segment-id segment-001 \
  --output /output/segment-001 --work-dir /workspace/segment-001 \
  --index /inputs/completed-index.json \
  --index-sha256 '<COMPLETED_INDEX_SHA256>' \
  --data-endpoint https://storage.givemeanode.com \
  --checkpoint-endpoint https://storage.givemeanode.com \
  --checkpoint-bucket '<PRIVATE_CHECKPOINT_BUCKET>' \
  --checkpoint-prefix training/training-run-20261023 \
  --checkpoint-every-steps 5000 --base-data-seed 42 \
  --streaming-smoke-report /inputs/streaming-benchmark-success.json \
  --exclusive-checkpoint-writer --acknowledge-stream-restart
```

`--temporal` selects `audioset_temporal_h100_four`: aggregation windows
`[1, 2, 2]`, supports `[1, 2, 4]`, and temporal-variance weight `0.1`, matching
its passing streaming acceptance report. Omitting the flag retains the pointwise
baseline, which requires its own matching report. Keep the same model selection
on every resumed segment; checkpoint metadata rejects a changed objective.

The first launch initializes an ownership marker in the empty prefix. To resume,
use the same run ID, prefix, bucket, base data seed, and frozen index; add
`--resume`, choose a new segment ID and fresh directories, and recheck the budget
before launching. The script discovers `latest.json`, downloads the entire
checkpoint, verifies its SHA256, and resumes through `Trainer.fit(ckpt_path=...)`.
A completed 375,000-step checkpoint is rejected as unnecessary work.

Clean mono global optimizer batch remains **64** (16 per GPU). Nat uses **16
recordings** (four per GPU), each with eight masked crops, for **128 masked
crops** per update. Both retain the absolute
**375,000-step** target, **100,000-step** LR warmup and existing **100,000-step**
EMA ramp. Optimizer moments, scheduler progress, EMA counters and Trainer loops
restore from the full checkpoint; only the official Lightning Timer's previous
elapsed time is reset for the new segment. Mismatched saved counters or schedules
are rejected. A segment must advance beyond its starting optimizer step before
it can report success.

For Nat, replace `--temporal` with `--nat` and supply
`--naturalistic-manifest /inputs/nat-manifest.json` and
`--naturalistic-manifest-sha256 '<SHA256>'`. The runner copies the exact manifest
into its working directory and binds its hash and crop recipe to checkpoint
ownership. Resuming with changed assets or switching between mono and Nat fails.
`scripts/start_nat_segment.py` can fetch the completed staging receipt and its
hash-verified manifest before invoking this runner; it does not rent GPUs.

### Optional clean-to-Nat curriculum

`--clean-binaural` selects `audioset_clean_binaural_h100_four`: clean AudioSet
is copied into both ears before chronological windows and the eight random
masked crops are drawn. The model, global batch of 16 recordings, crop sizes,
optimizer and absolute 375,000-step schedule match the Nat phase. This is an
explicit curriculum experiment, not Nat from the first update.

The clean segment can use `--stop-on-nat-ready-bucket` and
`--stop-on-nat-ready-key` to stop when the pinned staging receipt is complete.
Rank zero checks the small receipt every 100 optimizer steps; all ranks stop
together, validate and save a full checkpoint. Otherwise the ordinary segment
time budget stops the clean phase.

After the clean job succeeds, `scripts/wait_nat_curriculum.py` runs on a bounded
CPU job. It validates the clean handoff and full Nat manifest, then publishes a
small curriculum receipt. A dependent Nat job starts only after that gate
succeeds, preventing simultaneous GPU writers. If assets are late, waiting
consumes CPU time, not GPU time.

Nat imports the clean checkpoint into a **new owned checkpoint prefix** through
the explicit curriculum source options. The transition is bound to the verified
source checkpoint SHA, original data-state digest, frozen AudioSet index and
target Nat manifest. Only clean-binaural → naturalistic is accepted. Model
parameters, Adam state, LR scheduler, EMA and global step are restored; ordinary
resumes continue to reject changed data recipes. The shard iterator restarts
under the documented seed policy below.

Data iterator and prefetch positions are **not restored**. This limitation needs
the explicit acknowledgement flag. Each segment records the policy
`data_seed = base_data_seed + resumed_optimizer_step`, using the base seed for the
first segment. This changes the restarted shard/crop order without changing the
model seed or LR/EMA schedule; it does not provide sample-exact continuity.

## Durable handoff and failure behavior

The production runner first saves at update 250, then every 5,000 updates and at
a normal final stop. Lightning writes a full local checkpoint. Rank zero uploads it under an immutable step/SHA256 key, reads back
the **complete object**, verifies its SHA256 and byte count, then publishes and
verifies `latest.json`. All ranks wait for this result. A Timer stop captures the
final optimizer step before Lightning starts its final validation pass.

Rank zero prints compact progress every 100 optimizer steps, including scalar
loss and the observed interval update rate. A separate publication record appears
only after a checkpoint has passed complete remote verification.

Production shard reads share a 300-second total deadline across at most three
requests. Interrupted bodies resume the remaining byte range, requiring exact
range/length headers and a final whole-object SHA256 match. Worker batches have
a 360-second timeout, beneath the segment's minimum 720-second DDP timeout.
The ordinary non-segment presets retain their shorter timeout settings.

At most two checkpoint objects are retained beneath the owned prefix. Before
pruning the older object, the current verified checkpoint is reconfirmed; its
object remains while the replacement is uploaded and verified. Cleanup deletes
only validated checkpoint keys in that prefix and refuses foreign objects. This
is an exclusive-writer protocol, not a conditional-write lock.

Upload, readback, receipt, or ownership failure stops the run with a nonzero exit.
The previous durable checkpoint remains available; failure never triggers an
automatic fallback or restart. A hard kill may lose work since the last verified
checkpoint. Inspect `latest.json` and download verification before retrying after
an ambiguous upload result. There are no platform checkpoint POST/yield calls.

Loss curves and temporal diagnostics are flushed to small CSV files beneath
`<output>/metrics/training/`; cached audio and full checkpoints remain in the
separate working directory. The final receipt includes the model variant and
final scalar metrics.

Small local records are `segment-plan.json`, `latest-checkpoint.json`, optional
`resumed-checkpoint.json`, and, only after a verified advancing finish,
`segment-success.json`. The success record distinguishes `segment_finished`
from `training_complete`; only reaching step 375,000 completes training.

Local CPU tests cover full model/AdamW/LR/EMA resume, Timer reset, a durable save
before final validation, corruption and publication failures, bounded retention,
timeout configuration and rejection, two-rank Gloo publication success/error
handoffs, and watchdog termination during blocked setup. They do not establish live S3
checkpoint transfer time or corrected four-H100 streaming throughput.

# Train from staged AudioSet shards

The streaming input reads the WAV TAR files written by `src.stage_audioset` directly.
It uses [WebDataset](https://github.com/webdataset/webdataset) for TAR iteration and
sample shuffling, [boto3](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/get_object.html)
for private S3 reads, and PyTorch's data loader for worker processes and pinned-memory
batches. A thin cache layer verifies each object's indexed byte count and SHA256
before any audio is exposed to training.

Install the input dependencies with `uv sync --extra data`. Supply a bucket-scoped,
read-only storage key through `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the
job environment; credentials do not belong in Hydra overrides or configuration files.
The endpoint defaults to `https://storage.givemeanode.com` and region to `auto`.

Use the **completed** cohort's `index.json`, preferably copied to an immutable local
file in the training job. The loader also accepts its private `s3://bucket/key` URL.
Production rejects a partial index, a different source revision, overlapping source
objects, incorrect split assignments, and a falsely declared complete cohort.
Recording-ID disjointness across shards is established by the staging ledger; the
loader checks each shard's manifest checksum, identities, counts, and audio members.

```bash
uv run python -m src.train experiment=audioset_h100 data=sharded_audio \
  callbacks=step_checkpoint trainer.min_epochs=0 \
  data.index_path=/data/audioset-index.json \
  data.cache_dir=/data/audio-cache \
  data.batch_size=32 data.num_workers=4 \
  trainer.use_distributed_sampler=false
```

`batch_size` is per GPU: **32 × 2 GPUs** or **16 × 4 GPUs** gives global batch 64.
Keep `accumulate_grad_batches=1` for the same optimizer batch. The `audioset_h100`
experiment supplies the 9.09-second waveform, 375,000 optimizer updates, and an
explicit validation interval. Do not run this infinite training iterator using only
an epoch limit; set a positive `trainer.max_steps`.

Use `callbacks=step_checkpoint` for production streaming runs. It writes the
latest full model/optimizer/loop state every 5,000 optimizer updates, even when
validation does not improve, and explicitly saves the final state when training
ends normally between intervals. Retention is one latest step file plus a local
`last.ckpt` symlink. Step files are serialized to a temporary sibling, flushed,
and atomically renamed; the alias is also atomically replaced. The previous step
file is removed only after the new alias is published, so an interrupted save
leaves the previous `last.ckpt` usable. This guarantee applies to the local
`StepCheckpoint` callback; a host/disk loss still requires a durable copy.
SIGKILL can leave an orphan temporary or fully written step file, so successful-save
retention is not a crash-cleanup guarantee. Inspect owned leftovers before reuse.
A rank-zero serialization failure inside Lightning can leave peer ranks at its
save barrier until launcher teardown or the configured collective timeout; atomic
publication does not promise immediate cross-rank exception delivery.
An additional callback keeps one best validation checkpoint;
disable that optional file with `callbacks.best_checkpoint=null`. Checkpoint names
use update numbers, since this infinite iterator remains in epoch zero.

Set `trainer.min_epochs=0` so Lightning's official `trainer.max_time` / `Timer`
can stop an infinite run before its update budget. A platform kill or lost host
cannot promise a final save; keep a time margin for saving and durable upload.
The callback writes locally only. A production job must capture/upload the
regular step file (or follow `last.ckpt`), verify that durable capture succeeded,
and retain an older durable checkpoint until the replacement is verified. No
cloud upload or automatic recovery is supplied by this callback.

The default cache is capped at **40 GiB across the configured ranks and workers**,
including downloads in progress. It reserves separate allocations for training,
validation, and test readers. Every allocation must fit the largest shard; lower
the worker count if it cannot. A 2 GiB free-space floor protects the remainder of
the job disk. The directory belongs to one cohort/topology: changing the index,
GPU count, or worker count requires clearing that owned cache after the old job
stops, or selecting another empty directory. Old cache directories still occupy
disk until removed. Cache reuse rechecks SHA256, and a corrupt cached object is
downloaded again. Interrupted S3 response bodies retry the whole object up to three
times; checksum mismatches fail closed. `data.download_timeout_seconds=120` bounds
the complete GET/body/retry operation, including a slow response that keeps making
progress. This Unix signal deadline also protects synchronous held-out reads and
respects an earlier outer probe deadline. No complete corpus extraction is needed.

`data.worker_timeout_seconds=180` bounds the wait for a spawned worker batch.
Ordinary H100 runs set a separate 180-second collective timeout.
[Bounded training segments](TRAINING_SEGMENTS.md) extend it to cover synchronous
checkpoint publication plus 120 seconds (720 seconds by default). These are
failure bounds, not recovery: a stalled/failed rank aborts the job. A bounded
training segment additionally uses its existing parent-process watchdog to stop
all owned ranks and workers. Nonfinite updates remain fail-closed; optimizer/EMA
steps are not silently skipped. These changes have local CPU tests but still need
a fresh GPU streaming acceptance run.

Training assigns each shard to exactly one rank and worker. Each worker shuffles
its owned shards and a bounded sample buffer, cycles them indefinitely, and chooses
a deterministic random contiguous crop per recording and local pass. Every batch
is full; no short recording is padded. Workers can complete passes at different
times, so this stream has no synchronized whole-corpus epoch. The fixed ownership
means ranks/workers do not independently sample the same shard.

The default `data.shuffle_buffer=128` is **per worker**, against source shards of
roughly 2,000 clips. It mixes nearby samples in that worker's sequential shard
stream; it is not a corpus-wide shuffle or an IID sampler. Away from shard
transitions, one rank's batch is usually dominated by one source shard; with four
ranks the 64-recording regularizer population is then dominated by roughly four
source shards. Older buffered examples can cross a shard boundary. More workers
do not guarantee that a single global batch covers more sources, because PyTorch
assembles each rank's batch from one worker. The gathered regularizer population
inherits these correlations. Report worker count and shuffle-buffer size with
results and compare mixing sizes/seeds when evaluating regularization.

Larger `data.shuffle_buffer` values can mix across more adjacent shards but
increase initial loading and worker RAM: the buffer retains WAV bytes before
decoding, approximately 40 MB for 128 ten-second clips or 1.3 GB for 4,096 clips
per worker, plus Python/prefetch overhead. Larger buffers still do not establish
IID sampling. The cache's disk limit is separate from this RAM cost.

Validation and test read their own held-out source shards with deterministic crops.
They divide contiguous recording ranges across ranks, using one reader and the
same number of full batches per rank. Neighboring ranks may read the same boundary
shard, but their recordings are disjoint. Fewer than one global batch is omitted
from these **training-objective metrics** to keep collective regularizers aligned;
exact available, used, and omitted counts are exposed as
`datamodule.evaluation_coverage` and included in its checkpoint state.
This is not an exhaustive downstream evaluation; use the separate HEAR evaluation
for downstream results.

For a bounded throughput pilot only, use a frozen partial index and
`data.allow_partial_index=true`. Each rank/worker still needs at least one training
shard, so reduce `num_workers` for small pilots. A train-only pilot disables its
validation loader; it does not become a complete training cohort.

Checkpoint resume restores model/optimizer state, but WebDataset iterator positions
and prefetched batches are **not** checkpointed. By default the loader refuses a
resume. Explicitly set `data.allow_stream_restart_on_resume=true` to accept a new
iteration from cycle zero; the index SHA must still match. With an unchanged
configured seed, resume advances the previous effective stream seed by one and
records both `configured_seed` and `stream_seed` in checkpoint state. A deliberately
changed `data.seed` is used as supplied. This avoids exactly replaying the same
deterministic prefix, but previously seen recordings can still recur: it does not
recover sample positions or guarantee disjoint data across segments. Legacy
checkpoints without seed metadata advance the configured seed once.
The bounded segment runner explicitly uses `base_seed + resumed_global_step`;
that policy remains unchanged. None of these modes is bitwise-equivalent to an
uninterrupted run.

The fast presets sample training diagnostics every 50 updates (plus the first
step of each run); `train/*_epoch` therefore means an average of sampled batches.
Validation still evaluates every batch in the held-out loader. With an infinite
iterator the training epoch mean is normally finalized only when training ends;
use step metrics for live progress. Neither a lower diagnostic frequency nor a
checkpoint cadence changes the training loss or update budget.

Local tests exercise staged TAR compatibility, corruption handling, cache limits,
real worker processes, two CPU DDP ranks with collective validation, full batches,
deterministic crops, and resume guards. Real multi-GPU throughput and private S3
loading overlap still require a hardware benchmark.

## Bounded live input probe

`scripts/probe_sharded_audio.py` measures the production streaming input without
allocating a model or renting resources. Run it on a host that already has the
training dependencies; installing them on a busy preparation node is unnecessary.
Supply the same bucket-scoped read-only credentials and set
`HWAVJEPA_INDEX_BUCKET`, `HWAVJEPA_INDEX_KEY`, and `HWAVJEPA_S3_ENDPOINT` in the job
environment. The only remote operation is reading the index and selected objects.

```bash
python scripts/probe_sharded_audio.py \
  --allow-partial-index \
  --cache-parent /data \
  --output /data/sharded-input-probe.json
```

The probe freezes the fetched index SHA and selects at most two training shards,
with at most 1.5 GiB in its own temporary cache. It decodes at most 4,096 clips,
uses one CPU thread, and stops after 180 seconds. It removes its cache on success,
ordinary failure, SIGTERM, or its internal deadline. SIGKILL cannot run cleanup.
The optional partial-index flag is for this bounded pilot; production training
still requires the completed cohort.

Its JSON report includes the frozen index and source hashes, selected bytes, S3
GET/read timing, combined cache/hash verification timing, TAR iteration, PCM
decoding, and cold-stream clips per second. Cache timing includes network timing,
so these measurements must not all be added together. The report compares input
throughput with **134.4 clips/s** for 2.1 updates/s and **211.2 clips/s** for 3.3
updates/s at global batch 64. This comparison does not establish GPU compute speed
or distributed input scaling.

For a representative GPU input check, use a frozen partial index with at least
four nonempty training shards, two GPUs, two workers per rank, batch 32 per rank,
and a fresh bounded cache. Capture first-batch latency, then at least 256 measured
updates after warmup. Approximately 2,000 clips per shard means four readers need
roughly 125 updates to move beyond their initial shards; a shorter warmed benchmark
can miss ongoing S3 downloads. Disable validation for this throughput-only check
and retain a separate validation correctness test. Four GPUs with two workers each
need roughly 512 measured updates to exercise comparable shard turnover.

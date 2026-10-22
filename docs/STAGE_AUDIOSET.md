# Stage AudioSet on object storage with a small CPU disk

`src.stage_audioset` prepares and uploads one source shard at a time. It reuses
[the CPU preparation pipeline](PREPARE_AUDIOSET.md), so a 250 GiB CPU disk can
stage the complete corpus without retaining approximately 568 GB of WAVs locally.
The command uses Python, FFmpeg and the standard boto3 S3 client. It creates no
compute resources or credentials and rents no GPUs.

## Install and run a bounded pilot

On a minimal CPU host, install FFmpeg and `boto3>=1.43.118,<2` alongside Python 3.11
or newer. The staging modules do not import PyTorch. In a full project environment,
`uv sync --frozen --extra data` installs the optional S3 dependency.

Provide bucket-scoped credentials through `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` environment variables; `AWS_SESSION_TOKEN` is optional.
The endpoint is explicit and credentials never enter configuration files, object
keys, provenance or reports. Use a private bucket and a dedicated prefix.

For the SFCompute storage endpoint:

```bash
python3 -m src.stage_audioset \
  --output /data/audioset-staging \
  --bucket h-wavjepa-audioset \
  --endpoint https://storage.givemeanode.com \
  --prefix audioset \
  --source unbal_train000 \
  --max-sources 1 \
  --max-runtime-hours 1 \
  --max-upload-gib 2 \
  --max-local-gib 8 \
  --workers 2
```

`AWS_REGION` defaults to `auto`. The client uses path-style S3 addressing. To
reuse a previously prepared source, add `--reuse-prepared /data/audioset`.
Only a complete source with the expected pinned archive hash, preparation
contract, FFmpeg version and validated WAVs can be reused. Existing pilot files
remain intact; the stager hardlinks or copies them into its owned work directory.

The first invocation downloads and verifies the three original metadata CSVs,
about 104 MB, and uploads each once. It also pins the complete source catalog,
FFmpeg version and preparation-module hash in persistent staging state. All
870 unbalanced shards remain training data, balanced training is development
validation, and official evaluation is test data.

## Verification and deletion order

For each source, the stager:

1. Prepares it in an isolated work directory, verifies the entire original tar
   SHA256 and checks every accepted WAV's format, duration and checksum.
2. Checks recording IDs against the persistent registry of previously uploaded
   sources. A cross-split collision is fatal; a duplicate within one split is
   logged and excluded from the new archive.
3. Packages an uncompressed tar with WAVs, split manifests, clip provenance,
   rejection reasons and preparation reports. The original large FLAC tar is
   streamed and never retained.
4. Uploads to a content-addressed key containing the revision, source name and
   prepared tar's SHA256. It streams the entire object back and verifies its
   bytes and SHA256 independently of S3 user metadata.
5. Commits the source receipt and recording IDs to the persistent global ledger,
   publishes and verifies the remote dataset index, then deletes that source's
   owned local work directory.

Upload or checksum failures retain local WAVs for retry. A crash after upload but
before enrollment re-verifies the existing content-addressed object and enrolls
it exactly once. A crash after enrollment but before index publication keeps the
work directory until a resume successfully publishes the index.

SFCompute's tested gateway does not preserve user metadata or enforce conditional
PUT semantics. The stager therefore uses a unique cohort UUID subprefix, content
hashes, full readback and one writer per local staging state. It does not depend
on remote compare-and-swap. Initial object verification records the server ETag;
a later resume can use an unchanged ETag and size, and redoes full readback if
that evidence changes or is unavailable. Missing or corrupted objects stop the run.

## Resume and completion

Keep `/data/audioset-staging` on persistent CPU storage. Rerun with the **same
output directory, bucket, endpoint and prefix**. Preserve its SQLite ledger,
configuration and work directories; the registry is required for cohort-wide
recording-ID separation after earlier WAVs have been evicted.

The root index lives at `<prefix>/<cohort-id>/index.json` and records each verified
shard's object key, size, SHA256, split counts and manifest hashes. It also records
metadata objects and measured total payload bytes. Local `index.json` contains
the same information after successful publication. An interrupted final
publication remains explicitly partial.

A pilot or bounded run has `status: partial`. The index can say `complete` only
when all **889 source shards** and the three metadata objects have been verified.
A successful command does not imply a complete corpus: check the index status
and counts. Rejected short or invalid clips remain auditable in the shard reports.

An empty or missing ledger cannot silently restart against a populated remote
prefix. Existing ledgers are checked for integrity, cohort identity, recording
counts and agreement with the remote index. A stale backup that omits already
published shards is rejected. Restore the original state before continuing;
remote-only reconstruction of the registry is not implemented.

## Explicit budgets for a full cohort

After measuring the pilot and agreeing the CPU and storage budget, a bounded
full-cohort invocation can use:

```bash
python3 -m src.stage_audioset \
  --output /data/audioset-staging \
  --bucket h-wavjepa-audioset \
  --endpoint https://storage.givemeanode.com \
  --prefix audioset \
  --max-sources 889 \
  --max-runtime-hours 8 \
  --max-upload-gib 600 \
  --max-local-gib 8 \
  --workers 2
```

The runtime is a stopping budget, not a completion estimate. Resume with the same
state until the verified index is complete. Source count is a per-invocation
limit; the uploaded-byte budget applies cumulatively across resumes and includes
shards plus metadata. Small control/index objects are additional. Full source
transfer is about 2.46 TB, and prepared audio is approximately 568 GB before tar
headers; remote verification reads each newly uploaded object once more.

The working-space cap covers retained source work and packaging, with metadata
and the durable SQLite/index state stored separately. A source that cannot fit
is retained and reported as partial. Old owned tar packaging is removed before
repacking, so retries do not hold two full tar copies. Single-object uploads are
limited to 4 GiB. CPU preparation and metadata subprocesses receive the remaining
runtime deadline; termination can require five seconds, and an in-flight network
request can consume its bounded timeout/retry allowance before returning.

## Using the staged data

These objects are verified storage artifacts. The current training datamodule
still reads local WAV manifests and does not directly stream S3 tar shards. A
training host must either materialize the selected archives with sufficient local
disk or use a separately implemented and tested shard-streaming loader. Keep the
source index and checksums with the eventual training run. Staging alone does not
validate sustained two-H100 input throughput or downstream representation quality.

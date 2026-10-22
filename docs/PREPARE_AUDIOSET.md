# Prepare AudioSet before renting GPUs

The CPU-only preparation command uses the same pinned AudioSet research mirror
linked by WavJEPA. It streams the source archives, converts the existing clips
with FFmpeg, and writes the WAVs and JSONL manifests used by this repository.
It does not require PyTorch or a GPU. See [source provenance](AUDIOSET_SOURCE.md)
for the exact revision, available cohort and storage estimate.

All 870 unbalanced training shards remain assigned to training. The ten balanced
training shards provide development validation; the nine official evaluation
shards provide the test split. This development split is an explicit adaptation
of the original recipe. A recording ID may never cross these splits: a collision
stops preparation for review. Duplicate IDs within one split are logged and
excluded. The official CSV offsets describe the original YouTube video; the
archive audio is **already cut**, so those offsets are never applied again.

## CPU host and pilot

Use a persistent CPU host with Python 3.11 or newer and FFmpeg installed. Only
the Python standard library is needed for preparation. Transfer the repository,
or just `src/prepare_audioset.py` and `configs/data/audioset-source-catalog.json`. For a
standalone copy, run `python3 /path/prepare_audioset.py` and add
`--catalog /path/audioset-source-catalog.json` to each command below.

Prepare a small pilot with one source shard for each split:

```bash
python3 -m src.prepare_audioset --output /data/audioset \
  --source eval00 --source bal_train00 --source unbal_train000 \
  --max-clips 128 --workers 2
```

The cap limits **new conversion attempts per selected source per invocation**.
Short or invalid clips count toward the cap, so use the resulting manifest
counts when checking a pilot. Even a capped run streams each entire source
archive to check its pinned SHA256; these three shards transfer roughly 8.4 GB.
Source archives are not retained. Successful clips retain their full decoded
duration, normally ten seconds, so the training loader can choose random crops.

To measure complete-shard throughput, run the same command for
`--source unbal_train000` without `--max-clips`. Already processed clips are
skipped, and the rest of the shard is prepared. A complete, verified source is
skipped entirely on subsequent runs.

Retain the original metadata and validate the prepared output:

```bash
python3 -m src.prepare_audioset --output /data/audioset --metadata-only
python3 -m src.prepare_audioset --output /data/audioset --verify-only
```

The metadata command fetches the three pinned segment/label CSVs, about 104 MB,
and checks their SHA256 values. The verification command reads each prepared
WAV, checks its recorded checksum and rebuilds the manifests. Preparation pins
the full FFmpeg version on its first conversion invocation; changing that
version requires a separate output directory.

## Full preparation and storage

After checking the pilot, omit `--source` and `--max-clips` to prepare every
available source. Reserve at least **650 GB of local persistent disk** for the
approximately 568 GB of mono PCM16 audio, database, manifests and working space.
The default two workers use bounded input files of at most 32 MiB each; the
command stops before available disk reaches the configurable
`--minimum-free-gib` floor, which defaults to 1 GiB. It never pads a short clip.
Recordings below 145,440 real samples (9.09 seconds) or above 30 seconds after
conversion are rejected and listed in the report.

```bash
python3 -m src.prepare_audioset --output /data/audioset --workers 2
```

Full preparation transfers about 2.46 TB of source FLAC archives. A 250 GiB node
disk cannot hold the complete prepared corpus. For a smaller CPU host, use the
[rolling object-storage stager](STAGE_AUDIOSET.md), which uploads and verifies
one source at a time before removing its local WAVs. The command above retains
all WAVs locally. The training loader still requires local WAVs; object-storage
staging does not add streaming training. Do not interpret a successful pilot as
a completed dataset.

## Resume, reports and training preflight

Rerun the same command after interruption. SQLite records completed conversions,
source descriptors and per-clip checksums. Atomic WAV replacement and manifest
writes avoid publishing partial files. An interrupted source is re-streamed from
the beginning: processed members are skipped and its full SHA256 is checked
before new clips enter the manifests. Previously verified clips remain usable
if a later resume is interrupted. Archive checksum mismatches quarantine that
source and require investigation; do not bypass the check. Invalid audio and short recordings remain in the rejection log and are not
automatically retried. Filesystem/resource errors and FFmpeg timeouts interrupt
preparation instead; affected members remain eligible for the next resume.

Keep the output directory intact, including:

- `train.jsonl`, `val.jsonl`, `test.jsonl`: relative WAV paths and recording IDs.
- `preparation-report.json`: verified source hashes, split counts, samples,
  byte totals, manifest hashes and rejection counts.
- `clips.provenance.jsonl`: source archive/member, input/output SHA256 and sample
  count for every published recording.
- `rejections.jsonl`: invalid, short, duplicate or failed audio with its reason.
- `metadata/`: pinned original segment/label CSVs and their provenance.
- `preparation.sqlite3`, configuration and runtime files: resumable state and
  the FFmpeg version used for conversion.

The archive reader never extracts archive paths or links. It accepts only the
expected AudioSet audio path layout and eleven-character video IDs, and invokes
FFmpeg with forced WAV/FLAC demuxers and local file/pipe protocols.

Before GPU rental, run the existing experiment-aware validator in an environment
with the project dependencies installed:

```bash
uv run python -m src.validate_data --experiment audioset_h100 \
  --train /data/audioset/train.jsonl \
  --val /data/audioset/val.jsonl \
  --test /data/audioset/test.jsonl
```

It checks the actual training contract, including format, chronological length,
split separation and the minimum full batch: at least 64 training clips and four
clips in each held-out split for the default two-H100 configuration. Keep a
frozen copy of these manifests and reports with the eventual training run.

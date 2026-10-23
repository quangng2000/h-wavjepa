# AudioSet source and staging provenance

The preparation source is the **same Hugging Face snapshot referenced by the
official WavJEPA training configuration**, not the mutable current dataset branch:

- Dataset: `agkphysics/AudioSet`
- Revision: `5a2fa42a1506470d275a47ff8e1fdac5b364e6ef`
- Format: tar archives containing FLAC clips; most audio is 48 kHz/24 bit,
  with some 44.1 kHz/24 bit.
- Collection date stated by the mirror: March 2023.
- License declared by the mirror: CC BY 4.0.

Primary references:

- [Pinned WavJEPA AudioSet configuration](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/configs/data/audioset.yaml)
- [Official WavJEPA data instructions](https://github.com/labhamlet/wavjepa#getting-the-audioset-data)
- [Pinned dataset card](https://huggingface.co/datasets/agkphysics/AudioSet/blob/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/README.md)
- [Pinned archive inventory](https://huggingface.co/datasets/agkphysics/AudioSet/tree/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data)
- [Hub file sizes and SHA-256 inventory](https://huggingface.co/api/datasets/agkphysics/AudioSet/revision/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef?blobs=true)
- [Google's official AudioSet split definitions and metadata](https://research.google.com/audioset/download.html)

Google distributes segment metadata and precomputed features. The precomputed
features cannot replace raw waveforms for this model. The pinned audio mirror
is publicly accessible without a dataset access gate. The current mirror branch
uses a different Parquet layout and reports slightly different example counts;
it must not silently replace this revision.

## Project split policy

| Project split | Source archives | Expected source examples | Archive bytes |
| --- | --- | ---: | ---: |
| Train | `data/unbal_train000.tar`–`unbal_train869.tar` | 1,738,788 | 2,409,795,399,680 |
| Validation | `data/bal_train00.tar`–`bal_train09.tar` | 18,685 | 26,028,206,080 |
| Test | `data/eval00.tar`–`eval08.tar` | 17,142 | 23,774,750,720 |
| Total | 889 archives | 1,774,615 | 2,459,598,356,480 |

Expected examples come from the pinned dataset card; archive sizes come from
the Hub's file inventory. They are **not validated output counts**. Corrupt,
short, duplicate, or inconsistent clips must be reported during conversion.

Using every unbalanced shard preserves WavJEPA's declared pretraining cohort.
Using the official balanced-training subset as this project's validation set is
an explicit adaptation. The official evaluation subset remains held out for
testing. Do not copy the upstream README's overlapping train/validation shard
example. Enforce disjoint original video IDs across project splits.

## Identity and conversion

Archive members use paths such as
`audio/unbal_train/---1_cCGK4M.flac` and
`audio/eval/-1PZQg5Gi8A.flac`. The basename identifies the original YouTube video.
The accompanying CSVs contain video ID, start time, end time, and labels:

```text
data/unbalanced_train_segments.csv
data/balanced_train_segments.csv
data/eval_segments.csv
```

Audio members are already the extracted segments. **Do not apply the CSV's
original-video start offset a second time.** Decode the actual sample rate,
downmix to mono, resample to 16 kHz, and write PCM16 WAV. Preserve the full clip
so training can choose its chronological 9.09-second crop. Reject outputs shorter
than 145,440 real samples; never pad them into forecasting examples.

Training manifests contain loader-supported fields such as `path` and
`recording_id`. Keep source revision, shard SHA-256, member name, original time
range, conversion settings, output checksum, and rejection reasons in separate
provenance records; the loader rejects unknown manifest fields. Pin conversion
tool versions and verify each completed source archive against its catalog hash.

## Storage planning

The source archives total **2.460 TB decimal** (2,290.68 GiB), plus approximately
104 MB of metadata. A nominal ten-second clip requires 320,000 bytes of 16 kHz
mono PCM16 payload. Across the expected examples, WAVs with 44-byte headers would
occupy approximately **567.95 GB decimal / 528.95 GiB**. Actual durations,
headers, accepted counts, and filesystem overhead determine the final size.

Plan approximately **650 GB of usable storage** to materialize the converted
corpus with working headroom. A 250 GiB volume cannot hold it. Conversion can
process and upload one archive at a time instead of retaining all 2.46 TB of
originals: the largest source shard is under 2.87 GB. Account separately for
parallel workers, unfinished uploads, and checkpoints.

Object storage can hold prepared shard archives plus manifests and checksums.
The [sharded training loader](SHARDED_AUDIO.md) reads those private-S3 TARs with
verified, bounded local caching, without materializing the full corpus. The
separate `WAVManifestDataModule` still requires local WAV paths. A completed,
frozen index and passing GPU streaming validation remain prerequisites for the
full training run.

Source metadata, archive headers, and accessibility were inspected without
downloading the audio corpus. Availability and integrity of every full shard
are established only when preparation completes; no preparation duration is
claimed from metadata inspection.

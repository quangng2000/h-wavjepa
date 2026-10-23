# WavJEPA-Nat and the hierarchical adaptation

WavJEPA-Nat requires binaural naturalistic scenes. Duplicating clean mono audio
into two channels is an evaluation compatibility operation, not naturalistic
training. The existing clean AudioSet corpus supplies the sound sources; it does
not supply the room responses or background noise.

The Nat implementation is not ready for a production GPU run until the asset
and hardware gates below pass. The successful four-H100 **mono** temporal test
does not validate Nat throughput, memory use, scene generation, or checkpoint
compatibility. Its runtime and cost estimates must not be reused for Nat.

## Model and data recipe

The paper uses two independent waveform encoders, concatenated channel tokens,
time/channel positional embeddings, and matching temporal masks in both ears.
Its training scenes use 70,000 BRIR scene sets from 70 Matterport3D houses;
15,000 scenes from different houses are held out. Noise is localized in half
the scenes and diffuse in half. Localized scenes have one noise position;
diffuse scenes have three to five. WHAM! background noise is trimmed to at most
ten seconds, faded over 200 ms at each end, spatialized, and mixed at 5–40 dB
SNR. Context and EMA target branches operate on the same resulting scene.
[Paper, §3.3 and Appendix D](https://arxiv.org/html/2509.23238v1#S3.SS3).

The released Nat model confirms independent channel weights with
`in_channels: 2` and `share_weights_over_channels: false`. It uses 2.01-second
inputs at 16 kHz: **32,160 samples and 200 tokens per ear**, or 400 tokens after
concatenation. Preserve channel-major order: all left-ear tokens, then all
right-ear tokens. Each channel has its own CNN; the downstream Transformer is
shared across the concatenated sequence.
[Pinned model configuration](https://huggingface.co/labhamlet/wavjepa-nat-base/blob/15d95ff67fa98117b17e83a1653bbca97877ff6f/config.json),
[pinned Nat model](https://huggingface.co/labhamlet/wavjepa-nat-base/blob/15d95ff67fa98117b17e83a1653bbca97877ff6f/model.py).

Use `clean_data_ratio=0.0` as the default naturalistic recipe. The paper's clean
mixture ablation favors fully naturalistic training, and the accompanying GRAM
scene pipeline has this explicit default. Clean mixtures should be named
ablations. Preserve one normalization scale across both ears; independent
channel normalization changes interaural level differences.
[Paper, §4.3](https://arxiv.org/html/2509.23238v1#S4.SS3),
[GRAM naturalistic configuration](https://github.com/labhamlet/GRAM-T/blob/9d017173e9b405a86050d6d36aba3c01d02a8e6c/configs/data/audioset.yaml),
[WavJEPA crop normalization](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/wavjepa/jepa.py).

Render a continuous scene before drawing crops. Use the same room, source
position, noise configuration, and mixture across all windows of a hierarchical
example. Independently rendering successive windows would introduce artificial
temporal changes. Use a single scalar noise gain across both ears to preserve
spatial cues. The official GRAM helper implements this by summing energy over
channel and time before computing the SNR gain.
[GRAM scene mixing](https://github.com/labhamlet/GRAM-T/blob/9d017173e9b405a86050d6d36aba3c01d02a8e6c/src/data_modules/scene_module/generate_scenes_batch.py).

Two preprocessing conventions are explicit adaptations. The offline noise
converter takes the first channel of WHAM's stereo recording as the mono noise
source before applying scene BRIRs; the paper does not specify that channel
choice. The renderer measures SNR jointly over both ears and the full rendered
clip. GRAM's helper measures energy only within the dry noise's active interval.
For short noise, these definitions can produce different gains. Record these
choices in the experiment provenance and use them in every matched baseline;
neither is evidence of exact paper reproduction.

## Baseline versus hierarchical experiment

The masked Nat baseline draws **eight random 32,160-sample crops per scene**.
For an explicit global batch of 16 scenes, this produces 128 masked crops per
optimizer update; four GPUs receive four scenes each without accumulation.
Targets remain the normalized top-eight teacher-layer average. Preserve the
375,000-update schedule and 100,000-update learning-rate/EMA warm-up.

For H-WavJEPA-Nat, keep that masked branch and add a separate chronological
branch containing nine 16,160-sample windows from the same scene. The temporal
hierarchy aggregates with windows `[1, 2, 2]`, giving support `[1, 2, 4]`, and
retains its within-recording temporal penalty. These extra forward paths,
forecasting objectives, and readout are **our adaptation**, not part of the
WavJEPA-Nat paper. A matched hierarchy comparison must keep the scene sampler,
masked crops, batch convention, optimizer schedule, and data splits identical
between the masked-only, pointwise, and temporal arms.

All three arms currently share the already-staged AudioSet cohort, which keeps
clips with at least 9.09 seconds for the chronological branch. This also limits
the masked-only control's source population; it is another reason to call it a
matched Nat baseline rather than an exact reproduction of the paper's corpus.

Random masked crops may cover any time in a scene, as in ordinary JEPA
pretraining. They must not become visible inputs to a chronological predictor;
the forecasting branch still sees only its permitted prefix. With a rendered
scene shared by student and teacher, the Nat masked task is not clean-target
denoising or waveform reconstruction.

## Published source discrepancies

The public source is useful but is not a complete, unambiguous Nat training
recipe at the pinned WavJEPA revision:

- **Global batch.** The paper reports batch 16 and eight crops, giving 128.
  The released training script passes `trainer.batch_size` to a loader on each
  rank and prints `batch_size × samples_per_audio × num_gpus`. The clean preset
  uses batch 32 with two GPUs. Thus a paper-reported batch and the same YAML
  number can imply different global populations. Record global scenes, crops,
  per-rank batch, world size, and accumulation explicitly; do not claim exact
  reproduction from the YAML number alone.
- **Learning rate.** The paper's main text gives a peak of `2e-4`, while
  Appendix A and the released clean optimizer configuration give `4e-4`.
  State which convention a preset follows; the existing AudioSet adaptation
  follows the main-text peak.
- **Mask layout.** The channel extractor concatenates channel-major tokens.
  The released masker's channel branch rearranges to `(S C)`, interleaving
  temporal positions. Our binaural mask must instead repeat the full temporal
  mask in channel-major order, with a test that each left target has the same
  right target.
- **Scene helper.** The WavJEPA denoiser helper explicitly selects channel zero
  from both source and noise RIRs. It cannot generate a binaural Nat example
  unchanged. Its denoiser also uses a separate clean teacher and different
  optimization settings; this is not the Nat masked objective. GRAM's binaural
  scene helper preserves the channels and a common SNR gain.
- **Missing preset.** The pinned WavJEPA `train.py` registry selects the mono
  `ConvFeatureExtractor`; its `denoise` configuration targets a separate mono
  experiment. Having a channel extractor file in the repository does not make
  either entrypoint a working reproduction of Nat.

Sources: [training script](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/train.py),
[loader](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/data_modules/WebAudioDataModule.py),
[optimizer](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/configs/optimizer/adamW.yaml),
[channel extractor](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/wavjepa/extractors/audio_channel_feature_extractor.py),
[masker](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/wavjepa/masking.py),
[mono scene helper](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/data_modules/scene_module/generate_scenes_batch.py),
[denoiser](https://github.com/labhamlet/wavjepa/blob/6f4e1b1f18316c50768c8523f325ccffeaa6985b/wavjepa/denoiser.py).

## Required assets and capacity

The [metadata inventory](validation/wavjepa-nat-assets-2026-10-23.json) records
pinned revisions, object sizes and publisher hashes. It is **not** evidence of
completed downloads. Only a 2 KiB prefix of one BRIR shard was inspected.

| Asset | Verified metadata | Preparation needed |
| --- | --- | --- |
| [Official packed training BRIR scenes](https://huggingface.co/datasets/labhamlet/GRAM-Naturalistic-Scenes-Binaural/tree/546f796991721befaffce60152af59bfd9ad64ff/train/webdataset_output) | 70 TARs; 645,192,396,800 bytes. First member is float64 `(6, 2, 96000)`: one source and up to five padded noise RIRs. | Verify all source hashes and sample counts; convert with pinned, tested resampling; preserve scene identity and split provenance. |
| [Alternative original BRIR release](https://huggingface.co/datasets/labhamlet/BinauralRIRs/tree/77fe93dd3363bca8b6922ab8bb084fc264ca2fbe) | 31 raw TARs; approximately 337.45 GB, plus 70 train and 15 test house metadata files. | Resolve individual source/noise RIR references using metadata; do not sample held-out houses for training. |
| [WHAM! noise corpus](http://wham.whisper.ai/) | Publisher ZIP is 18,194,930,748 bytes; published MD5 `d5af15645d521d3920e01954c6cd7594`. Training split: 20,000 files, 58.03 hours, 16 kHz stereo float32. | Verify ZIP, retain the training split, use the documented noise-source channel policy, resample only if necessary, and retain attribution. |

The packed BRIR data is declared 32 kHz by the official GRAM configuration.
If all 70,000 scenes have the inspected shape, preserving all three seconds,
resampling to 16 kHz and storing float32 would require **161.28 GB of array
payload** before headers or compression. This is a capacity estimate, not a
measured finished dataset. Removing zero-padded unused noise slots and lossless
compression may reduce it; truncating RIRs is a recipe change that needs its own
justification. Stream one source shard at a time during conversion instead of
retaining the 645 GB source corpus alongside converted output.

Both BRIR repositories declare MIT in their dataset metadata. The WHAM! noise
publisher specifies **CC BY-NC 4.0**, with attribution and noncommercial terms.
The repository's MIT code license does not relicense training audio. Only the
noise corpus is needed here; the WSJ0 access requirement on the WHAM page
concerns constructing their speech-separation mixtures. Generating fresh
Matterport3D scenes would instead require the relevant simulator, mesh access,
and dataset terms. Reusing published BRIRs avoids downloading the house meshes.
[BRIR metadata](https://huggingface.co/datasets/labhamlet/GRAM-Naturalistic-Scenes-Binaural/blob/546f796991721befaffce60152af59bfd9ad64ff/README.md),
[noise publisher and terms](http://wham.whisper.ai/).

The separately released [NatHEAR benchmark](https://huggingface.co/datasets/labhamlet/NatHEAR)
is evaluation data. It must not substitute for the training BRIR scene set.

## Offline asset conversion

`src.prepare_nat_assets` converts already-downloaded publisher assets. It never
downloads data, uploads objects, or starts paid compute. Install the optional
Nat dependencies first, and use a new output directory for each attempt:

```bash
uv sync --extra nat
python -m src.prepare_nat_assets brir \
  --source /data/source/shard-000000.tar \
  --expected-sha256 742c0107c75f680117818160b8dcb554c35e2871c4d27ec14e47376776977930 \
  --source-member train/webdataset_output/shard-000000.tar \
  --output-dir /data/nat/converted/shard-000000 \
  --max-items 1000 --max-output-bytes 3221225472 --max-seconds 3600
```

The example hash is the pinned inventory's first BRIR shard. Select the matching
inventory hash for every other shard; do not hash an unverified download and
treat that value as publisher verification. For WHAM, verify the archive against
the publisher's MD5 in the inventory, then record its SHA-256 and supply it to
the analogous `wham --source ... --expected-sha256 ... --output-dir ...` command.

The BRIR converter reads one bounded NPY member at a time, rejects links and
unsafe paths, and checks the NPY shape before allocating. It preserves complete
three-second responses and uses `scipy.signal.resample_poly` for antialiased
32-to-16 kHz resampling. Each scene becomes one compressed float32 NPZ containing
`arr_0[source, noise_1, ...]`; only exactly zero padded noise rows are dropped.
The scene's source and noise stay paired. The noise converter reads only WHAM's
`tr/` split, retains each clip's duration, and writes mono PCM16 with a recorded
common peak gain if needed to avoid quantization clipping. It does not perform
the later scene-specific crop or fade.

Each conversion emits an incremental `manifest.json` with relative
content-addressed object keys, byte sizes, SHA-256 values and packed-array
indices. Output objects are published atomically. A bound or failure preserves
completed objects and marks the manifest `partial` or `failed`; a fresh output
directory is required for another attempt. The byte limit bounds completed
asset payloads; allow additional space for one source member, temporary output
and manifest metadata. These are **component manifests**: they need verification,
upload, and assembly with the bucket, endpoint, complete BRIR population and
training noise population before they become a production training manifest.

Assemble the completed local components into an immutable remote manifest and
an explicit upload plan:

```bash
python -m src.prepare_nat_assets assemble \
  --brir-manifest /data/nat/converted/shard-*/manifest.json \
  --noise-manifest /data/nat/converted/wham-tr/manifest.json \
  --endpoint https://storage.givemeanode.com \
  --bucket h-wavjepa-audioset --prefix nat-assets/v1 \
  --output-manifest /data/nat/manifests/train-v1.json \
  --upload-plan /data/nat/manifests/train-v1-upload.json \
  --max-output-bytes 274877906944 --max-seconds 10800
```

Assembly rechecks every local object's size and SHA-256, validates paired scene
indices and unique references, rejects partial or wrong-split components, and
checks the BRIR source receipts against the pinned inventory. Full-corpus mode
requires 70 distinct training shards with 1,000 scenes each and 20,000 training
noise files. `--allow-pilot` permits a subset for testing, explicitly marks its
coverage as `pilot`, and does not establish full-corpus readiness. Pilot mode
still requires both localized and diffuse scenes and verifies all supplied
local objects. Existing output manifests and upload plans are never replaced.

The offline assembler requires all converted objects to remain available until
assembly and upload verification finish. Size the CPU staging disk for the
converted corpus plus one source archive and scratch space; the bounded GPU
cache is a separate mechanism and does not remove this preparation requirement.

The staged layout is deterministic:

```text
s3://h-wavjepa-audioset/nat-assets/v1/
  objects/<sha256>.npz   # One packed, matched BRIR scene; several logical indices
  objects/<sha256>.wav   # One converted training noise file
  manifest.json
```

The upload plan maps each **verified local file** to its exact remote key,
size, and hash. Upload and full-readback-verify the objects before publishing
`manifest.json`, then retain the remote verification receipt separately. The
assembly command itself performs **no uploads, downloads, or remote checks**;
its manifest says `remote_verified=false` and its plan reports
`offline_verified_not_uploaded`. This output can instantiate the renderer's
remote-metadata path, but cannot prove that S3 contains the files or that a GPU
run can read them. A remote readback and the Nat-specific hardware test remain
required.

## Configurations, weights and hardware verification

Install with `uv sync --frozen --extra data --extra nat --extra hear`.
The main presets are `audioset_nat_h100_four`,
`audioset_nat_pointwise_h100_four` and `audioset_nat_baseline_h100_four`.
Each requires both `data.index_path` and `data.naturalistic_manifest`.
The Nat manifest must use the same private bucket and endpoint as AudioSet;
the loader uses that bucket's read-only credentials for both. A separate cache
budget bounds Nat objects, divided among ranks and consumers. Worker setup and
checkpoint resume bind the exact manifest bytes and masked-crop recipe.

Render one full scene before drawing the eight random masked crops and the
chronological excerpt. Silent or DC-only sources keep their sample positions:
the zero-source limit gives zero noise gain and a zero mixture, with undefined
SNR recorded in rendering metadata. Invalid/no-energy noise assets still fail.
This avoids dropping different numbers of examples on different ranks.

`model.backbone.pretrained=true` selects the pinned official Nat safetensors for
these two-channel presets. For explicit preparation on a machine with room for
the approximately 801 MB file:

```bash
uv run python -m src.pretrained --variant nat --cache-dir /data/model-cache
```

The downloader verifies the published SHA-256 and never executes Hub Python.
Local tests check all 465 published tensor names/shapes, position-byte hashes,
and strict loading using small fixtures. They do **not** establish that the
complete released weight file has already been downloaded or loaded here.
The [HEAR API](HEAR.md) accepts native stereo for Nat checkpoints and explicitly
duplicates mono HEAR input into both ears for compatibility.

Once real assets are staged, the bounded hardware probe entrypoint is:

```bash
export HWAVJEPA_INDEX_BUCKET=h-wavjepa-audioset
export HWAVJEPA_INDEX_KEY='<completed-audioset-index-key>'
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/benchmark_streaming_gpu.py --nat --devices 4 --workers 1 \
  --naturalistic-manifest /inputs/nat-train-v1.json \
  --steps 600 --warmup 25 \
  --output /output/nat-streaming --scratch-parent /workspace \
  --checkpoint-bucket h-wavjepa-checkpoints \
  --checkpoint-prefix probes/nat-unique-run --checkpoint-run-id nat-unique-run \
  --exclusive-checkpoint-writer
```

Inject AudioSet read-only and separate checkpoint credentials through the job's
secret environment as described in [bounded segments](TRAINING_SEGMENTS.md).
The command itself neither rents hardware nor imposes a provider cost cap; a
separate authorized job must bound runtime, cost and restarts. Nat uses 600
updates by default because four recordings per rank need more updates to cross
a full AudioSet shard. Acceptance requires new remote BRIR **and** noise
objects after warm-up on every rank, finite updates to both independent ear
CNNs, full AudioSet validation apart from the global-batch tail, and durable
checkpoint recovery. This validates held-out audio clips rendered with training
rooms; it is not held-out-house or NatHEAR evaluation.

The production segment runner currently rejects Nat reports until a Nat launch
contract and budget have been established from measured hardware results.

## Gates before renting production GPUs

1. Stage verified training BRIRs and noise with an immutable manifest, hashes,
   sample rates, scene identities, split labels, and license provenance. Use
   synthetic fixtures only for tests, never as a silent production fallback.
2. Check rendering on real assets: nonidentical channels, correct crop alignment,
   common SNR gain, finite samples, localized/diffuse counts, and stable scene
   parameters across chronological windows. Verify held-out houses cannot enter
   the training sampler.
3. Run the actual four-rank Nat model with the intended global batch and both
   masked and chronological branches. Require every rank to refill the real
   AudioSet, BRIR and noise caches, finite parameter updates, held-out validation,
   and durable checkpoint upload/download/load. Check that both channel encoders
   receive gradients and that checkpoint metadata rejects a mono/Nat mismatch.
4. Measure scene generation, input stalls, GPU memory, optimizer updates per
   second, validation, and checkpoint transfer time. Recompute the bounded
   segment budget from this Nat measurement before production training.

A small asset subset can validate implementation, but cannot establish complete
70,000-scene preparation or full-corpus throughput. Until these gates pass, the
production status is **Nat code implemented; assets and GPU acceptance
pending**.

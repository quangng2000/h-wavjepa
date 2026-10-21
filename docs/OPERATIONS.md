# Operational contract

This is an executable research training/evaluation/extraction workflow. Production readiness still requires workload-specific acceptance tests for deployed reliability, resources and audio representation quality.

## Safeguards

- Preflight rejects incompatible clip/patch/mask settings before model allocation.
- WAV validation checks splits, payload sizes, physical aliases and optional group/recording separation.
- Nonfinite waveform inputs and losses fail explicitly. For unscaled precision, invalid gradients fail before AdamW/EMA mutation; mixed-precision GradScaler retains its standard overflow-skip behavior.
- Evaluation masks use local RNG independent of training RNG; repeatability requires fixed batch order/size and shard layout.
- Versioned model checkpoints reject semantic mismatches even when tensor shapes match.
- Checkpoint restoration explicitly requests weights-only loading. Files/configs must still be trusted: restricted loading does not bound resource use or make hostile inputs safe.
- Application artifacts reject reused run directories. Hydra's own logging/config handling precedes this guard; do not deliberately reuse `hydra.run.dir`.
- Outcome/metric JSON records are atomically replaced locally. Embedding shards publish after complete writes; failed directories remain for diagnosis.

## Artifacts

| File | Meaning |
| --- | --- |
| `run.json` | Running/completed/failed/interrupted status; timestamps, step/device/precision and checkpoint paths/hashes |
| `metrics.json` | Finite final scalar objectives, not a downstream benchmark |
| `config.yaml` | Resolved Hydra configuration |
| `provenance.json` | Versions, Git revision/dirty flag, seed, parameters, input checkpoint and manifest hashes |
| `csv/version_0/metrics.csv` | Step/epoch logs when CSV is enabled |
| `checkpoints/` | Lightning best/last checkpoints when enabled |
| `.hydra/`, `config_tree.log`, `tags.log` | Hydra/template records |

Only `completed` denotes normal return. Hard termination/power loss may leave `running`; it is not success. Manifest hashes cover manifest bytes, not the audio corpus. Keep input files immutable.

Configs/hyperparameters can expose secrets if you put them there. Use SDK/environment credential stores and review artifacts before sharing. External logging services are disabled by default.

## Checkpoint boundary

Checkpoints contain `hwavjepa.schema_version=1`, semantic identifiers, encoder construction settings and module hyperparameters. Mismatches in pooling, widths, masking, protection, prediction weights, EMA decay, evaluation seed or sample rate fail. Inference reconstructs these fields instead of executing saved configuration.

Old unversioned checkpoints are **not automatically migrated**: tensor shapes cannot recover pooling/objective choices. Preserve their original configs and use the old revision, or perform a separately audited migration.

Lightning restores optimizer/loop state, but exact random-stream replay and dataset-position continuation are not guaranteed. Optimizer class/config, split identity, seed, clip length and Trainer settings are outside the model metadata contract; check the original recorded config. Resume into a new output directory.

## Validation and remaining gates

Tests cover CPU model/data contracts, generated signals in real WAV containers through training/evaluation/extraction, EMA, resume, evaluation repeatability, failed-run records, output collisions, CLI entry points, Hydra sweeps and a small Optuna search. These fixtures are software tests, not real-world quality benchmarks. CI is configured for Python 3.11–3.13.

On 20 October 2026, a one-training-batch/one-validation-batch smoke run also passed on this machine's Apple MPS GPU with strict deterministic algorithms, Python 3.12.13, PyTorch 2.14.1 and Lightning 2.6.6. Best/last checkpoints and a completed step-1 run record were verified. The masked mean uses a dense reduction with tested equivalent gradients, avoiding MPS's unsupported deterministic scatter-add backward. This small check is not long-run, mixed-precision, resume or multi-device validation.

Spawn/fork/notebook distributed strategies are explicitly rejected: a two-process CPU diagnostic found Lightning's parent Trainer retains stale step/checkpoint metadata after workers exit. Single-device execution is the supported tested path. Externally launched DDP and accelerator/precision configs still require independent validation; inherited presets are not certification.

Before a production deployment:

1. Validate the actual hardware, distributed launcher and precision setup.
2. Measure peak memory, throughput, long-run stability and interruption recovery on the intended corpus.
3. Establish recording/factor-disjoint downstream tasks, matched baselines, collapse diagnostics and uncertainty estimates.
4. Validate artifact access, storage capacity, retention and monitoring.

No pretrained quality, official WavJEPA reproduction, serving SLA, exact RNG continuation or universal generalization guarantee is claimed. The companion [Lean proofs](https://github.com/quangng2000/hjepa-lean) are conditional mathematical results, not verification of this nonlinear implementation.

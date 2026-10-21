# Theory-to-code obligations

The authoritative theorem ledger is [hjepa-lean](https://github.com/quangng2000/hjepa-lean/blob/main/THEORY.md). This repository contains ordinary Python/PyTorch, not extracted or formally verified code.

| Formal idea | Scaffold implementation | Remaining obligation |
| --- | --- | --- |
| Deterministic three-level abstraction can discard task signal. | Aligned local pooling/projection at each level. | No claim these learned features improve task risk. Compare the same accessible context and matched probes. |
| Retaining a protected coordinate at all levels preserves its availability. | The same fine-resolution protected tensor appears in every level's readout bundle. | Availability is not successful readout learning; pooling the protected stream would change this condition. |
| Exact reconstruction of a genuinely fixed reference preserves all its readouts. | Raw waveform patches are a fixed target; a separate encoder/decoder minimizes reconstruction MSE. | Empirical approximate reconstruction is not population equality; approximate task guarantees require sensitivity assumptions. Raw-reference quality is not semantic quality. |
| Positive protection weight plus feasibility and approximate optimality can bound task-information loss. | Configurable reconstruction weight and explicit diagnostics. | This objective is not the labeled task objective, and neither feasibility nor neural optimization quality is certified. |
| Specified full-batch/SGD updates learn tiny protected linear encoders. | AdamW trains a nonlinear model with learned decoder and EMA predictive branch. | These are different algorithms/parameterizations. No transferred convergence theorem. Joint training can have bad stationary points. |

## Information-flow contract

1. Patch windows do not overlap. No waveform-wide normalization mixes hidden and visible samples.
2. A raw-patch mask is applied before the online predictive encoder; pooled masks mark any group containing a hidden patch.
3. Masked level tokens are replaced before the predictor consumes them. Target encoders may see the full input and are frozen/EMA-updated, not gradient-trained.
4. The unmasked protection branch has separate parameters and no connection into masked predictors. Its loss shares the optimizer, not its inputs or activations.
5. Perturbing hidden samples must leave student predictions unchanged for a fixed mask and weights. Tests check this at all three levels.

This implementation uses one aligned block mask across scales. It does not reproduce WavJEPA's released architecture, target normalization or masking recipe; it does not implement causal forecasting, H-JEPA planning, or SIGReg. A future global frontend, overlapping convolution, skip connection, normalization change, or pretrained integration requires a fresh support/leakage audit.

## Evidence we do not yet have

- Non-collapse and useful semantics after sustained training.
- H-WavJEPA versus WavJEPA downstream superiority.
- Matched-parameter/compute architectural benefit.
- Subject/recording/factor-disjoint generalization on real audio.
- HEAR probes, effective-rank/PCA studies, confidence intervals, or confirmatory seed runs.
- Bitwise resume, multi-device behavior, mixed-precision robustness, or production readiness.

Small tests establish software behavior only. Loss curves and passing tests must not be presented as proofs of these claims.

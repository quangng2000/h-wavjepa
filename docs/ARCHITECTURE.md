# Architecture

This page describes the earlier independent prototype (`hwavjepa`/`audio`).
The official-WavJEPA-backed route is documented in [Upstream dynamics](UPSTREAM_DYNAMICS.md).

## Library boundary

PyTorch supplies tensors, autograd, layers, Transformer predictors, data loading and AdamW. Lightning owns training, precision/device handling, accumulation and checkpoints. Hydra/Optuna compose and search configurations. The actual Lightning–Hydra template supplies orchestration and utilities. Project code implements the hierarchy, masking, protected path, objective, data contracts and small operational guards—not a replacement framework.

## Predictive path

Audio is divided into non-overlapping patches. Patch-local learned projections produce low-level tokens; aligned mean pooling and learned projections form coarser levels. Each level has its own position-aware bidirectional PyTorch Transformer predictor. A frozen EMA encoder supplies targets.

Hidden raw patches are zeroed **before** student encoding. Group-aligned masks propagate through pooling so hidden raw support cannot enter visible coarse context. The loss sums weighted latent MSE over hidden tokens. This is noncausal masked prediction, not forecasting.

EMA updates follow actual optimizer steps rather than accumulation microbatches. The teacher remains in evaluation mode; checkpoints include its weights and update counter.

## Protected path

A separate branch encodes and reconstructs unmasked raw patches, a genuinely fixed reference. Fine-resolution protected tokens are carried unchanged in every level's readout bundle, never fed to masked predictors. The objective adds optional raw-reference reconstruction MSE.

A smaller protected width is a bottleneck; reconstruction is not generally exact. Raw reconstruction may preserve nuisance information and increase cost. It does not automatically imply semantic abstraction.

Readouts are structured bundles, not automatically aligned concatenations. A downstream probe must choose alignment/pooling and account for additional dimensions. Mean pooling in the embedding CLI can discard information.

## Interpretation limits

Low/medium/high denotes temporal hierarchy, not frequency or loudness. Synthetic fixtures are only 64 ms long with 2/4/8 ms local token supports. The audio preset provides larger context but is not a validated representation design.

Predictive losses in separately learned spaces are not comparable task scores. Target/prediction standard deviations are diagnostics, not anti-collapse proofs; averaging local statistics across devices does not produce global statistics. EMA does not prove anti-collapse, and Lean results do not prove convergence of this nonlinear AdamW/EMA system. See [theory boundaries](THEORY_BOUNDARY.md).

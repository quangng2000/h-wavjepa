# Pinned upstream WavJEPA core

Source: https://github.com/labhamlet/wavjepa
Commit: `6f4e1b1f18316c50768c8523f325ccffeaa6985b`.
Original file SHA-256 values and modifications are recorded in `UPSTREAM.json`.

The original root license is retained as `LICENSE` (BSD-3-Clause, PierreGtch).
This package does not import upstream denoising, data loading, ARCH, or HEAR
evaluation packages. It is an attributed source integration, not an independent
reimplementation and not an endorsed upstream release.

## Third-party notice

The previously copied AudioMAE positional utility has been removed from active
code. `pos_embed.py` now implements only the mathematical 1D sinusoidal formula,
independently, with the sine-then-cosine channel ordering required by WavJEPA.
Float32 outputs are checked against golden hashes generated from the unmodified
pinned upstream implementation. The original source hash and `LICENSE-AudioMAE`
remain solely as historical provenance for earlier repository revisions. This
change is not a statement about commercial rights to pretrained weights, other
dependencies, or historical revisions containing the removed source.

## Integration boundary

The adapter calls the official `JEPA.forward` and masked loss. It replaces the
original random-crop/shuffle transfer hook with chronological, independent-window
normalization and does not use the upstream optimizer or training-step hooks.
The outer Lightning module owns optimizer scheduling and post-step EMA timing.
Only the context Transformer is the official EMA teacher: frontend, feature norm
and projection are shared with the student, then detached for teacher targets.
The teacher is kept in evaluation mode explicitly.

Deterministic masking threads an optional local NumPy Generator through the
original masking algorithms; retry counts are bounded to avoid an infinite loop.
This does not seed or replace global random functions. All other masked-loss and
target-normalization equations are retained, including upstream's actual
four-dimensional `instance_norm` behavior.

No upstream `train.py`, data loader or global CUDA configuration is executed.
Model weights are not bundled. `checkpoint_path` loads a local original training
state dictionary; `pretrained`/`pretrained_path` explicitly select the official
Hugging Face export layout described below. Evaluation dependencies live in the
separate HEAR environment.

## Exact sample counts and pretrained exports

The JEPA constructor additionally accepts an optional integer `target_length`.
The adapter passes its explicit window size, avoiding the upstream floating-point
round trip that truncates `2.01 * 16000` to 32159 samples. Omission retains the
original constructor behavior; learned tensors and positional token counts are
unchanged for the released 200-token base model.

The official Hugging Face base release is loaded as safetensors from
`labhamlet/wavjepa-base` revision
`6be4a5093f6c9adbbd55e00b6e5b8f067aa03345`. Its complete tensor mapping
(student, teacher, frontend and decoder) is preserved after removing only the
explicit `model.` prefix. No Hub Python is executed. The downloadable artifact
is SHA-256 checked against
`988546976c453353c14ebb0798f275ea5eb95270d75eed506ea455c5b49f6be0`.

The original `encode_tokens` and legacy window readout retain the final encoder
output. New dynamics configurations explicitly select `normalized_top_k`, which
uses matching student and teacher layer-selection/normalization before averaging.
Existing checkpoints default to the legacy readout. The masked branch exposes
its already-computed detached teacher means for reuse by forecasting.

Gradient checkpointing now preserves attention padding masks in the student
encoder, covers the decoder, and checkpoints the new online top-k traversal.
Target normalization and masked-loss reductions run in FP32 during mixed
precision. The default FP32 masked objective remains numerically unchanged.
The adapter continues to reject `compile_modules`.

The official Hugging Face release can initialize a shorter-window model by
regenerating only the two fixed sinusoidal positional tensors. Both source
tensors must exactly match the original 200-token base grid; all learned tensors
still require exact names, shapes and dtypes. The AudioSet preset uses 16160
samples (1.01 seconds, 100 tokens), an explicit adaptation of the released
2.01-second window, not an identical reproduction of its training recipe.

Token readout exposes contextual encoder output before averaging.
Its nominal centers follow the convolution strides and receptive fields; its
context remains the entire completed window, not only that convolution support.

## Optional masked-token compaction

`compact_masked_tokens: true` gathers visible positions before the student
encoder and each target-conditioned decoder, pads only to the longest visible
row in the current batch, and scatters results back to their original positions.
The positional embeddings are added before gathering. Masked keys cannot affect
visible queries, and discarded query outputs do not enter the masked loss, so
this preserves the zero-dropout objective while avoiding unused query work.
The dense teacher traversal, all learned weights, and state-dict names remain
unchanged. Both traversals use the original PyTorch Transformer modules (which
already use scaled-dot-product attention when attention weights are not needed).

The flag defaults to false for previous configurations. Compaction does not
promise bitwise agreement: attention reductions can change floating-point order.
Training with nonzero dropout is rejected because packing would change dropout
sampling. Tests compare visible predictions, teacher targets, losses, all
parameter gradients, checkpointing, BF16 loss behavior, and checkpoint loading
against the dense path. CPU correctness does not establish an H100 speedup.

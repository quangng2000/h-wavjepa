# Pinned upstream WavJEPA core

Source: https://github.com/labhamlet/wavjepa
Commit: `6f4e1b1f18316c50768c8523f325ccffeaa6985b`.
Original file SHA-256 values and modifications are recorded in `UPSTREAM.json`.

The original root license is retained as `LICENSE` (BSD-3-Clause, PierreGtch).
This package does not import upstream denoising, data loading, ARCH, or HEAR
evaluation packages. It is an attributed source integration, not an independent
reimplementation and not an endorsed upstream release.

## Third-party notice

`pos_embed.py` retains Meta's copyright notice and explicitly cites
https://github.com/facebookresearch/AudioMAE/blob/main/util/pos_embed.py.
AudioMAE publishes an Attribution-NonCommercial 4.0 license, retained here as
`LICENSE-AudioMAE` from https://github.com/facebookresearch/AudioMAE/blob/main/LICENSE.
The exact provenance of every positional utility is not independently resolved;
do not assume WavJEPA's root BSD license overrides those inherited restrictions
or that this integrated bundle is cleared for commercial use.

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

Token readout exposes the original contextual encoder output before averaging.
Its nominal centers follow the convolution strides and receptive fields; its
context remains the entire completed window, not only that convolution support.

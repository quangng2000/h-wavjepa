# Template provenance

This scaffold follows the configuration groups and `src/data`, `src/models`, `src/utils`, train/eval separation of [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

Inspected upstream revision: `bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144` (20 October 2026).

This is a compact, independently authored adaptation of that layout, **not a full fork or a verbatim copy of its MNIST implementation**. The upstream checkout is not vendored. It does not inherit upstream Git history or include unused MNIST/download/logger integrations.

Differences:

- `uv` project and lockfile instead of multiple environment installers.
- Module entry points (`python -m src.train` / `src.eval`) from the project root; no `sys.path` mutation.
- Local CSV logs, explicit opt-in test evaluation, and saved provenance.
- Waveform hierarchy, raw-mask support propagation, EMA targets, and a separate fixed-reference protection branch.
- Tiny deterministic fixtures and local WAV manifests, with no implicit downloads.
- CI configured for CPU tests; hardware presets remain opt-in.

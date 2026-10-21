# Actual Lightning–Hydra template integration

Source: [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

Pinned revision: [`bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144`](https://github.com/ashleve/lightning-hydra-template/tree/bddbc24b82ab6ccfa6243e815a49dc5bfe8d4144), imported on 20 October 2026.

The upstream MIT copyright and permission notice is retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The initial H-WavJEPA commit followed the template's conventions with independently written orchestration. This migration **imports the upstream source**, then applies explicit audio/project customizations. It does not merely keep similar directory names. Existing H-WavJEPA Git history is retained; no repository recreation, force-push, or claim of GitHub-generated template ancestry is needed.

## Imported and retained

| Upstream component | Treatment |
| --- | --- |
| `src/train.py`, `src/eval.py` | Imported upstream pipeline and API: `rootutils`, rank-aware logging, task wrapper, extras, Hydra instantiation, hyperparameter logging, metric/object dictionaries, optimized-metric return. Project-specific safety and provenance additions listed below. |
| `src/utils/instantiators.py`, `logging_utils.py`, `pylogger.py`, `rich_utils.py`, `utils.py` | Imported upstream utility source, retained unchanged. |
| `src/utils/__init__.py` | Upstream exports plus explicit `__all__` and the project provenance helper. |
| `configs/train.yaml`, `eval.yaml` | Upstream composition structure, with H-WavJEPA/synthetic/CSV/CPU defaults and explicit experiment/debug/local settings. |
| `configs/callbacks`, `debug`, `extras`, `logger`, `trainer`, `paths`, `hydra` | Imported upstream groups; audio metrics, current Lightning options and tiny-fixture compatibility adaptations. Optional external logger presets remain opt-in. |
| `configs/hparams_search` | Upstream Optuna recipe adapted to `hwavjepa_optuna.yaml`, optimizing validation loss over optimizer parameters/batch size. |
| `.env.example`, `.pre-commit-config.yaml`, `Makefile`, `requirements.txt`, `notebooks/` | Template development workflow customized for this project and its locked uv/Ruff toolchain. |

## H-WavJEPA retained intact

The migration does not change `src/models/`, `src/data/`, or `src/objectives/`: hierarchy, protected stream, mask propagation, predictors, EMA update mechanism, synthetic fixtures, WAV manifests, and objectives stay as implemented. Their leakage, shape, data-split, and teacher tests remain. Additional integration tests exercise the upstream pipeline.

## Explicit customizations

- Preserve `uv.lock` and Python 3.12 rather than adopting the template's old Conda/Python-version matrix. No unnecessary torchvision or MNIST downloads.
- Keep CSV logging and `test=false` by default; do not activate remote services. Logger presets need their separate SDKs/accounts before use. Use `logger=[]` to disable loggers.
- Correct seed-zero handling; preserve a configurable CPU thread count for small fixtures.
- Retain `config.yaml` / `provenance.json` via `src/utils/provenance.py`. Remove the earlier competing `runtime.py` orchestration and custom root resolver.
- Support both direct-script and module entry points through upstream `rootutils` and `.project-root`.
- Validate local checkpoint paths, retain safe loading, support callback/logger-free training, and require a checkpoint for test-only execution.
- Use `val/loss` for checkpointing, early stopping, and Optuna; none of these is a cross-model downstream research score.
- Adapt debug batch limits to tiny audio fixtures and remove MNIST-only datamodule fields. Keep microsecond run-directory timestamps to reduce collisions.
- Use the upstream pre-commit hygiene checks with locked Ruff rather than conflicting old formatter stacks. Keep the project's lightweight CPU GitHub workflow; do not activate upstream release-drafting or Codecov publishing integrations.

## Not included or not claimed

MNIST example classes/configs/tests are replaced, not retained as dead sample code. The generic upstream `setup.py` placeholder package and Conda installer are superseded by this project's uv/pyproject setup. Optional external logger services, distributed accelerators, and production performance are not validated by CPU smoke tests. This template migration does not expand the scope of the companion Lean theorems.

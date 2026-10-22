# Archived independent prototype

This namespace contains the original local-pooling waveform prototype and its
protected raw-patch reconstruction objective. It is retained for old checkpoint
loading and historical comparisons; new research development uses
`src/models/hierarchical_wavjepa_module.py` and the official WavJEPA backbone.

The prototype experiment configurations live under `configs/experiment/legacy`:
`experiment=legacy/smoke`, `legacy/audio`, `legacy/hwavjepa`,
`legacy/hwavjepa_unprotected`, `legacy/flat`, `legacy/flat_protected`, and
`legacy/two_level`. Their previous short names remain compatibility aliases.
The model configuration retains its previous `_target_` spelling for saved Hydra
configurations and existing runtime dispatch; that module now re-exports this
implementation. Old imports refer to the same class/function objects, and
versioned state-dictionary checkpoint semantics are unchanged.

The feature-extraction CLI defaults to the active dynamics model. Select
`--model-family legacy` for this prototype. The existing Python
`extract_embeddings` API retains its legacy default for compatibility.

See [historical usage](../../docs/USAGE.md) and
[the prototype's theory boundary](../../docs/THEORY_BOUNDARY.md).

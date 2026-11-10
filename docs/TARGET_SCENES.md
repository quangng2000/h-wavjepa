# Independent target scenes

`data.independent_target_scene=true` renders every recording in **two**
independently drawn Nat scenes. The student still sees only the first. Levels
selected by `model.independent_target_levels` take their future EMA targets
from the second. This is an untested research option. It is off by default, and
existing presets, data states and checkpoints are unchanged.

## What changes

The default pipeline renders one scene per recording and shares it between the
context, the masked crops and every future target. With the option on:

- The decoder renders the same dry recording a second time with its own seed.
  Room, source position, noise file, noise placement and SNR are drawn again.
  The same chronological excerpt is returned as `target_waveform`.
- The EMA teacher encodes `target_waveform` exactly as it encodes `waveform`.
  For each flagged level, the future target is read from that encoding.
- Unflagged levels, the online context, the masked anchor and its crops keep
  the first scene.

The whole recording is rendered once per scene before any excerpt is taken.
Successive windows are never rendered separately, so no scene change occurs
inside a context or inside a target.

The presets flag the medium and high levels and leave the low level shared:

```yaml
data:
  independent_target_scene: true
model:
  independent_target_levels: [false, true, true]
```

`validate_config` rejects a model and a data setting that disagree, and the
option without a naturalistic manifest. A checkpoint records both settings;
resuming with a different one is rejected.

## Why try it

All future targets otherwise come from the recording and scene the context came
from. Anything constant within that scene is then perfectly predictable, which
is the recording-identity shortcut described in
[temporal diagnostics](TEMPORAL_DIAGNOSTICS.md). H-JEPA counters the analogous
problem with inverse-dynamics supervision, which needs actions that audio lacks.

[Balestriero and LeCun](https://arxiv.org/abs/2610.39302) argue that a JEPA
learns what its two views share, and that views whose nuisance is drawn
independently yield a function of the underlying state. Redrawing the scene for
the target removes room, position and noise from what the context can predict.

The companion repository holds machine-checked statements of this mechanism for
a two-sign toy model in its unreviewed tier:
[`FUZZ_QUEUE.md`](https://github.com/quangng2000/hjepa-lean/blob/716165e32cc3d70207d6c1e927e3acff0a97c56d/FUZZ_QUEUE.md).
They are not claims about this network, its optimizer, or audio. In that model
a longer horizon never changes which factor is selected, so horizons 1, 2 and 4
alone do not make the levels differ; a redrawn factor is not selected.

## What it costs

- **Spatial information at flagged levels.** A level cannot learn to predict
  what was redrawn for its target. The toy-model statement is that every linear
  readout of the redrawn factor then fails. Keep at least one level and the
  masked anchor on the shared scene if room or position matters downstream.
- **Rendering.** Scene rendering doubles for every recording.
- **Teacher compute.** One additional no-gradient teacher pass per step: all
  windows with temporal aggregation, three windows for the pointwise model.
- **What stays shared.** Microphone, codec and any other property of the
  original AudioSet recording appear in both scenes. Redrawing cannot remove
  them.

Throughput and memory with the option on are unmeasured.

## Diagnostics with two scenes

Forecast origins stay in the context scene. `persistence_mse` therefore
compares a context-scene state with a second-scene target and includes the
change of scene. `forecast_gain_over_persistence` is larger for that reason
alone. Do not compare either metric between a shared-scene arm and an
independent-scene arm. Within-recording and between-recording variance are
computed from context states only and keep their meaning.

## The 2×2 comparison

| | Masked anchor | No masked anchor |
| --- | --- | --- |
| Shared scene | `audioset_nat` | `audioset_nat_future_only` |
| Independent upper targets | `audioset_nat_independent_targets` | `audioset_nat_independent_targets_future_only` |

Each has a `_h100_four` form. A test checks that every arm equals its baseline
except for tags, the two target-scene settings, `masked_weight` and the masked
crop recipe. The no-anchor arms set `masked_weight: 0` and disable separate
masked crops, so their update sees 16 chronological recordings and no crops.

Report, per level and per arm: `within_variance_fraction`, absolute within and
between variance, and downstream scene, event and spatial probes. A spatial
probe is needed to see the cost above.

## Masked-weight sweep

The 2×2 only compares a masked anchor at weight 1 with none. The toy-model
statement is finer: a masked term keeps content only above some weight, and
only if content has more energy than nuisance in the masked target; with
independent targets, too heavy a masked term can undo their effect. Neither
threshold is known for audio, so sweep the weight in both scene arms:

| Scene arm | Masked weight 0 | 0.3 | 1 | 3 |
| --- | --- | --- | --- | --- |
| Shared | `audioset_nat_future_only_h100_four` | `audioset_nat_h100_four model.masked_weight=0.3` | `audioset_nat_h100_four` | `audioset_nat_h100_four model.masked_weight=3.0` |
| Independent upper targets | `audioset_nat_independent_targets_future_only_h100_four` | `audioset_nat_independent_targets_h100_four model.masked_weight=0.3` | `audioset_nat_independent_targets_h100_four` | `audioset_nat_independent_targets_h100_four model.masked_weight=3.0` |

Each cell is the value of `experiment=`, followed by its override where one is
shown. The grid is a starting point, not derived from any measurement. Weight 0
is a separate preset because it also removes the separate masked crops. A test
checks that the override changes nothing but the weight.

## Horizon check

In the toy model a longer horizon never changes which factor is selected; it
only weakens the guarantee that independent targets give. To test whether the
horizon matters here, compare a matched pair that differs only in the horizons
and the excerpt length they require:

| Horizons | Command after `experiment=audioset_nat_independent_targets_h100_four` |
| --- | --- |
| 1, 2, 4 | `model.aggregation_windows=[1,1,1]` |
| 1, 2, 3 | `model.aggregation_windows=[1,1,1] model.horizons=[1,2,3] data.num_samples=129280` |

Both use the **pointwise** hierarchy, and 1, 2, 3 is the only shorter set
available:

- Horizons must be strictly increasing.
- With temporal aggregation the medium and high targets span 2 and 4 windows.
  Horizons 2 and 4 are therefore already the shortest for which every target
  lies after the observed prefix. A shorter horizon there would put part of
  the target inside the prefix. `validate_config` does not check this, so do
  not shorten horizons on an aggregated preset.

A test checks that every Nat arm preset keeps its targets after the prefix and
that the pair above differs only in horizons and excerpt length.

## Boundaries

- Nat only. Mono and clean-binaural data have no scene to redraw, and the
  option is rejected there.
- No result here shows the option helps. The four arms above are the test.
- `scripts/benchmark_streaming_gpu.py` and the segment runners were neither
  changed nor exercised with the option on.

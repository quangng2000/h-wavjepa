"""Independent target scenes: data rendering, target selection and config guards."""

from copy import deepcopy
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_nat_integration import nat_batch, nat_model
from test_naturalistic_audio import _assets, _with_checksums
from test_sharded_audio import make_wav, module
from torch.utils.data import DataLoader

from src.data.sharded_audio_datamodule import _decode
from src.utils.runtime import validate_config

UPPER = [False, True, True]
MANIFEST = "/explicit/nat-manifest.json"
ARM_DATA_KEYS = ("independent_target_scene", "masked_crops_per_recording", "masked_crop_samples")
ARM_MODEL_KEYS = (
    "independent_target_levels",
    "masked_weight",
    "masked_crops_per_recording",
    "masked_crop_samples",
)
POINTWISE = "model.aggregation_windows=[1,1,1]"
SHORT_HORIZONS = ("model.horizons=[1,2,3]", "data.num_samples=129280")
ARMS = {
    "audioset_nat_independent_targets": (True, 1.0, 8),
    "audioset_nat_independent_targets_h100_four": (True, 1.0, 8),
    "audioset_nat_future_only": (False, 0.0, 0),
    "audioset_nat_future_only_h100_four": (False, 0.0, 0),
    "audioset_nat_independent_targets_future_only": (True, 0.0, 0),
    "audioset_nat_independent_targets_future_only_h100_four": (True, 0.0, 0),
}


def renderer(audio, *, generator):
    shift = torch.rand((), generator=generator)
    return torch.stack([audio + shift, audio * 0.3 - shift])


def decode(**changes):
    options = dict(
        raw=make_wav((torch.sin(torch.arange(4096) * 0.1) * 20000).int().tolist()),
        name="clip.wav",
        sample_rate=16000,
        num_samples=1152,
        seed=42,
        cycle=0,
        random_crop=True,
        naturalistic=renderer,
        crops_per_recording=8,
        crop_samples=256,
    )
    return _decode(**(options | changes))


def nat_experiment(experiment, *overrides, manifest=MANIFEST):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        return compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "data.index_path=/explicit/index.json",
                f"data.naturalistic_manifest={manifest}",
                *overrides,
            ],
        )


def without_arm_settings(cfg):
    plain = OmegaConf.to_container(cfg, resolve=False)
    plain.pop("tags")
    for section, keys in (("data", ARM_DATA_KEYS), ("model", ARM_MODEL_KEYS)):
        for key in keys:
            plain[section].pop(key)
    return plain


def as_plain(cfg):
    return OmegaConf.to_container(cfg, resolve=False)


def target_supports(cfg):
    supports = [1]
    for window, stride in zip(
        cfg.model.aggregation_windows[1:], cfg.model.state_strides[:-1], strict=True
    ):
        supports.append(supports[-1] + (window - 1) * stride)
    return supports


def test_decode_adds_a_second_scene_without_changing_the_first():
    shared = decode()
    first = decode(independent_target_scene=True)
    again = decode(independent_target_scene=True)
    changed = decode(independent_target_scene=True, cycle=1)
    assert "target_waveform" not in shared
    for key in shared:
        torch.testing.assert_close(first[key], shared[key], atol=0, rtol=0)
    assert first["target_waveform"].shape == first["waveform"].shape == (2, 1152)
    assert not torch.equal(first["target_waveform"], first["waveform"])
    torch.testing.assert_close(first["target_waveform"], again["target_waveform"], atol=0, rtol=0)
    assert not torch.equal(first["target_waveform"], changed["target_waveform"])
    # This renderer only adds a per-scene offset, so a constant difference
    # shows both scenes hold the same excerpt of the same dry recording.
    difference = first["target_waveform"][0] - first["waveform"][0]
    assert difference.abs().max() > 0
    assert difference.std() < 1e-6


def test_decode_rejects_a_second_scene_without_rendering():
    with pytest.raises(ValueError, match="requires naturalistic rendering"):
        decode(naturalistic=None, independent_target_scene=True)


def test_datamodule_requires_assets_and_binds_the_option_on_resume(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requires a naturalistic manifest"):
        module(tmp_path, monkeypatch, independent_target_scene=True)
    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    options = dict(naturalistic_manifest=str(path), allow_stream_restart_on_resume=True)
    data, _ = module(tmp_path, monkeypatch, independent_target_scene=True, **options)
    iterator = iter(data.train_dataloader())
    batch = next(iterator)
    del iterator
    assert batch["target_waveform"].shape == batch["waveform"].shape == (2, 2, 8)
    assert torch.isfinite(batch["target_waveform"]).all()
    assert not torch.equal(batch["target_waveform"], batch["waveform"])
    state = data.state_dict()
    assert state["independent_target_scene"] is True
    data.load_state_dict(state)
    shared, _ = module(tmp_path, monkeypatch, **options)
    assert "independent_target_scene" not in shared.state_dict()
    with pytest.raises(ValueError, match="target-scene rendering changed"):
        shared._check_resume(state)
    with pytest.raises(ValueError, match="target-scene rendering changed"):
        data._check_resume(shared.state_dict())


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"aggregation_windows": [1, 1, 1], "temporal_variance_weight": 0.0},
        {"masked_crops_per_recording": 0, "masked_crop_samples": None},
        {"masked_weight": 0.0, "masked_crops_per_recording": 0, "masked_crop_samples": None},
    ],
    ids=["aggregated", "pointwise", "window-anchor", "future-only"],
)
def test_only_flagged_levels_take_targets_from_the_second_scene(changes):
    torch.manual_seed(0)
    model = nat_model(independent_target_levels=UPPER, **changes).eval()
    plain = nat_model(**changes).eval()
    plain.load_state_dict(model.state_dict())
    batch = nat_batch()
    if not model.masked_crops_per_recording:
        batch.pop("masked_crops")
    target = torch.randn_like(batch["waveform"])
    with torch.no_grad():
        mixed = model.compute_loss(**batch, target_waveform=target, seed=7)
        context = plain.compute_loss(**batch, seed=7)
        other = plain.compute_loss(**(batch | {"waveform": target}), seed=7)
    torch.testing.assert_close(mixed["targets"][0], context["targets"][0])
    for level in (1, 2):
        torch.testing.assert_close(mixed["targets"][level], other["targets"][level])
        assert not torch.allclose(mixed["targets"][level], context["targets"][level])
    # The student never sees the second scene.
    for got, expected in zip(mixed["predictions"], context["predictions"], strict=True):
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
    torch.testing.assert_close(mixed["prediction_terms"][0], context["prediction_terms"][0])


def test_second_scene_reaches_only_the_flagged_levels_losses():
    torch.manual_seed(0)
    model = nat_model(independent_target_levels=UPPER).eval()
    batch = nat_batch()
    targets = [torch.randn_like(batch["waveform"]) for _ in range(2)]
    first, second = (
        model.compute_loss(**batch, target_waveform=target, seed=7) for target in targets
    )
    torch.testing.assert_close(first["prediction_terms"][0], second["prediction_terms"][0])
    for level in (1, 2):
        assert first["prediction_terms"][level] != second["prediction_terms"][level]
    first["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan", "unexpected"])
def test_target_waveform_must_match_the_model_recipe(damage):
    levels = [False, False, False] if damage == "unexpected" else UPPER
    model, batch = nat_model(independent_target_levels=levels), nat_batch()
    target = torch.randn_like(batch["waveform"])
    if damage == "missing":
        target = None
    if damage == "shape":
        target = target[..., :-128]
    if damage == "dtype":
        target = target.double()
    if damage == "nan":
        target[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="waveform|crop"):
        model.compute_loss(**batch, target_waveform=target)


@pytest.mark.parametrize(
    "changes",
    [
        {"independent_target_levels": [True, True]},
        {"independent_target_levels": [0, 1, 1]},
        {
            "independent_target_levels": UPPER,
            "prediction_weights": [1, 0, 1],
            "temporal_variance_weight": 0.0,
        },
    ],
    ids=["length", "type", "inactive-level"],
)
def test_independent_levels_are_validated(changes):
    with pytest.raises(ValueError, match="independent"):
        nat_model(**changes)


def test_checkpoints_bind_independent_levels_and_default_old_metadata():
    plain, flagged = nat_model(), nat_model(independent_target_levels=UPPER)
    old = {plain.CHECKPOINT_KEY: plain.checkpoint_metadata()}
    del old[plain.CHECKPOINT_KEY]["dynamics"]["independent_target_levels"]
    plain.on_load_checkpoint(deepcopy(old))
    with pytest.raises(ValueError, match="configuration mismatch"):
        flagged.on_load_checkpoint(deepcopy(old))
    flagged.on_load_checkpoint({flagged.CHECKPOINT_KEY: flagged.checkpoint_metadata()})


def test_lightning_trains_and_validates_with_a_second_scene():
    torch.set_num_threads(1)
    batch = nat_batch(4) | {"target_waveform": torch.randn(4, 2, 9 * 128)}
    rows = [{key: value[i] for key, value in batch.items()} for i in range(4)]
    loader = DataLoader(rows, batch_size=2)
    model = nat_model(independent_target_levels=UPPER)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        max_steps=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader, loader)
    assert trainer.global_step == int(model.ema_updates) == 2
    assert torch.isfinite(trainer.callback_metrics["val/loss"])


@pytest.mark.parametrize("experiment", ARMS)
def test_experiment_arms_validate_and_differ_only_in_views_and_anchor(experiment):
    independent, masked_weight, crops = ARMS[experiment]
    cfg = nat_experiment(experiment)
    validate_config(cfg)
    assert cfg.data.independent_target_scene is independent
    assert list(cfg.model.independent_target_levels) == (UPPER if independent else [False] * 3)
    assert cfg.model.masked_weight == masked_weight
    assert cfg.data.masked_crops_per_recording == cfg.model.masked_crops_per_recording == crops
    assert cfg.data.batch_size * cfg.trainer.devices == 16
    baseline = "audioset_nat_h100_four" if experiment.endswith("h100_four") else "audioset_nat"
    assert without_arm_settings(cfg) == without_arm_settings(nat_experiment(baseline))


@pytest.mark.parametrize(
    "overrides, manifest, message",
    [
        (["data.independent_target_scene=false"], MANIFEST, "must agree"),
        (["model.independent_target_levels=[false,false,false]"], MANIFEST, "must agree"),
        (["model.independent_target_levels=[false,true]"], MANIFEST, "three booleans"),
        ([], "null", "requires a naturalistic manifest"),
    ],
    ids=["data-off", "model-off", "length", "no-assets"],
)
def test_runtime_rejects_inconsistent_target_scene_options(overrides, manifest, message):
    cfg = nat_experiment("audioset_nat_independent_targets", *overrides, manifest=manifest)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


@pytest.mark.parametrize("weight", [0.3, 3.0])
@pytest.mark.parametrize(
    "experiment", ["audioset_nat_h100_four", "audioset_nat_independent_targets_h100_four"]
)
def test_masked_weight_sweep_changes_only_the_weight(experiment, weight):
    cfg = nat_experiment(experiment, f"model.masked_weight={weight}")
    validate_config(cfg)
    swept, baseline = as_plain(cfg), as_plain(nat_experiment(experiment))
    assert swept["model"].pop("masked_weight") == weight
    assert baseline["model"].pop("masked_weight") == 1.0
    assert swept == baseline


@pytest.mark.parametrize("experiment", ["audioset_nat", "audioset_nat_h100_four", *ARMS])
def test_every_arm_keeps_its_targets_after_the_observed_prefix(experiment):
    cfg = nat_experiment(experiment)
    assert all(
        horizon >= support
        for horizon, support in zip(cfg.model.horizons, target_supports(cfg), strict=True)
    )


def test_short_horizon_pair_is_pointwise_and_differs_only_in_horizon():
    experiment = "audioset_nat_independent_targets_h100_four"
    default = nat_experiment(experiment, POINTWISE)
    short = nat_experiment(experiment, POINTWISE, *SHORT_HORIZONS)
    for cfg in (default, short):
        validate_config(cfg)
        assert target_supports(cfg) == [1, 1, 1]
    assert list(default.model.horizons) == [1, 2, 4]
    assert list(short.model.horizons) == [1, 2, 3]
    window = short.model.backbone.config.window_samples
    assert short.data.num_samples == window * (short.model.context_windows + 3)
    plain_default, plain_short = as_plain(default), as_plain(short)
    for plain in (plain_default, plain_short):
        plain["model"].pop("horizons")
        plain["data"].pop("num_samples")
    assert plain_default == plain_short

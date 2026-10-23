"""Nat data/model contracts with analytic audio; no corpus or GPU downloads."""

import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from test_nat_backbone import nat_config
from test_naturalistic_audio import _assets, _remote_assets, _with_checksums
from test_sharded_audio import ENDPOINT, make_wav, module
from torch.utils.data import DataLoader

from src.data.naturalistic_audio import NaturalisticAudio
from src.data.sharded_audio_datamodule import _decode
from src.dynamics_inference import load_model
from src.embed import extract_embeddings
from src.hear_api import HEARModel, get_scene_embeddings, get_timestamp_embeddings
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule
from src.utils.runtime import validate_config


def nat_model(**changes):
    options = dict(
        backbone=WavJEPABackbone(nat_config()),
        optimizer=partial(torch.optim.AdamW, lr=1e-3),
        hidden_dim=16,
        predictor_layers=1,
        predictor_heads=2,
        sigreg_projections=8,
        readout_mode="normalized_top_k",
        aggregation_windows=[1, 2, 2],
        temporal_variance_weight=0.1,
        masked_crops_per_recording=8,
        masked_crop_samples=256,
    )
    return HierarchicalWavJEPAModule(**(options | changes))


def nat_batch(batch_size=2):
    return {
        "waveform": torch.randn(batch_size, 2, 9 * 128),
        "masked_crops": torch.randn(batch_size, 8, 2, 256),
    }


def test_full_scene_render_precedes_every_crop_and_is_reproducible():
    raw = make_wav((torch.sin(torch.arange(4096) * 0.1) * 20000).int().tolist())
    calls = []

    def renderer(audio, *, generator):
        calls.append(len(audio))
        shift = torch.rand((), generator=generator)
        return torch.stack([audio + shift, audio * 0.3 - shift])

    kwargs = dict(
        raw=raw,
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
    first = _decode(**kwargs)
    again = _decode(**kwargs)
    changed = _decode(**(kwargs | {"cycle": 1}))
    assert calls == [4096] * 3
    assert first["waveform"].shape == (2, 1152)
    assert first["masked_crops"].shape == (8, 2, 256)
    for key in first:
        torch.testing.assert_close(first[key], again[key], atol=0, rtol=0)
        assert not torch.equal(first[key], changed[key])


def test_nat_hierarchy_loss_uses_independent_masked_crops_and_updates_both_ears():
    model = nat_model()
    batch = nat_batch()
    result = model.compute_loss(**batch, seed=7)
    assert torch.isfinite(result["loss"])
    assert [len(x) for x in result["predictions"]] == [2, 2, 2]
    direct = model.backbone.masked_loss(batch["masked_crops"], seed=7)
    torch.testing.assert_close(result["masked_loss"], direct)
    result["loss"].backward()
    for ear in model.backbone.model.extract_audio.cnns:
        assert sum(p.grad.abs().sum() for p in ear.parameters() if p.grad is not None) > 0
    assert model.backbone.model.pos_encoding_encoder.shape[1] == 62
    prefix = batch["waveform"][..., : 5 * 128]
    assert [x.shape for x in model.predict(prefix)] == [(2, 16)] * 3


def test_nat_masked_only_control_does_not_forecast_or_use_chronological_view():
    model = nat_model(
        hierarchy=False,
        prediction_weights=[0, 0, 0],
        regularization_weight=0,
        temporal_variance_weight=0,
        aggregation_windows=[1, 1, 1],
    )
    batch = nat_batch()
    first = model.compute_loss(**batch, seed=7)
    second = model.compute_loss(**(batch | {"waveform": torch.randn(2, 2, 400)}), seed=7)
    torch.testing.assert_close(first["loss"], first["masked_loss"])
    torch.testing.assert_close(first["loss"], second["loss"])
    assert first["predictions"] == []


@pytest.mark.parametrize("damage", ["missing", "count", "length", "channels", "nan"])
def test_nat_rejects_missing_or_wrong_crop_recipe(damage):
    model, batch = nat_model(), nat_batch()
    if damage == "missing":
        batch.pop("masked_crops")
    if damage == "count":
        batch["masked_crops"] = batch["masked_crops"][:, :7]
    if damage == "length":
        batch["masked_crops"] = batch["masked_crops"][..., :-1]
    if damage == "channels":
        batch["waveform"] = batch["waveform"][:, 0]
    if damage == "nan":
        batch["masked_crops"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="waveform|crop"):
        model.compute_loss(**batch)


def test_nat_real_lightning_updates_and_checkpoint_restore(tmp_path):
    torch.set_num_threads(1)
    batch = nat_batch(4)
    rows = [{k: v[i] for k, v in batch.items()} for i in range(4)]
    loader = DataLoader(rows, batch_size=2)
    model = nat_model()
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
    path = tmp_path / "nat.ckpt"
    trainer.save_checkpoint(path)
    restored = load_model(path)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], atol=0, rtol=0)
    # The mono manifest exporter declares its two-ear adaptation and agrees with
    # a direct Nat inference call at the same chronological grid.
    audio = tmp_path / "clip.wav"
    audio.write_bytes(make_wav(range(1152)))
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps({"path": str(audio), "recording_id": "clip"}) + "\n")
    output = tmp_path / "embeddings"
    record = extract_embeddings(
        path, manifest, output, num_samples=1152, batch_size=1, model_family="dynamics"
    )
    assert record["input_channels"] == 2
    assert record["input_adaptation"] == "mono duplicated to both ears"
    exported = torch.load(output / "batch-000000.pt", weights_only=True)["levels"]
    with torch.inference_mode():
        encoded = restored.encode(
            (torch.arange(1152).float() / 32768)[None, None].expand(1, 2, -1), subsample=True
        )
    for got, expected in zip(exported, encoded, strict=True):
        torch.testing.assert_close(got, expected.mean(dim=1))
    saved = torch.load(path, weights_only=True)
    changed = deepcopy(saved)
    changed[model.CHECKPOINT_KEY]["dynamics"]["masked_crops_per_recording"] = 7
    with pytest.raises(ValueError, match="configuration mismatch"):
        model.on_load_checkpoint(changed)


@pytest.mark.parametrize("representation", ["low", "medium", "high", "backbone"])
def test_nat_hear_preserves_stereo_and_explicitly_duplicates_mono(representation):
    model = HEARModel(nat_model(), representation=representation, window_batch_size=2)
    mono = torch.randn(2, 800).tanh()
    stereo = mono[:, None].repeat(1, 2, 1)
    torch.testing.assert_close(
        get_scene_embeddings(mono, model), get_scene_embeddings(stereo, model)
    )
    values, times = get_timestamp_embeddings(stereo, model)
    assert values.shape[:2] == times.shape
    assert torch.all(times[:, 1:] > times[:, :-1])
    assert times.max() <= 800 / 16
    asymmetric = stereo.clone()
    asymmetric[:, 1] = asymmetric[:, 1].roll(10, dims=-1)
    assert not torch.equal(
        get_scene_embeddings(stereo, model), get_scene_embeddings(asymmetric, model)
    )


@pytest.mark.parametrize(
    "experiment",
    [
        "audioset_nat",
        "audioset_nat_h100_four",
        "audioset_nat_baseline",
        "audioset_nat_baseline_h100_four",
        "audioset_nat_pointwise",
        "audioset_nat_pointwise_h100_four",
    ],
)
def test_nat_configs_have_explicit_assets_and_matching_paper_crop_batch(experiment):
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "data.index_path=/explicit/index.json",
                "data.naturalistic_manifest=/explicit/nat-manifest.json",
            ],
        )
    validate_config(cfg)
    assert cfg.model.backbone.config.input_channels == 2
    assert cfg.data.batch_size * cfg.trainer.devices == 16
    assert cfg.data.masked_crops_per_recording == cfg.model.masked_crops_per_recording == 8
    assert cfg.data.masked_crop_samples == cfg.model.masked_crop_samples == 32160
    assert cfg.model.lr_total_steps == 375000
    assert cfg.model.lr_warmup_steps == 100000


def test_nat_streaming_renders_real_assets_and_binds_resume_recipe(tmp_path, monkeypatch):
    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    data, _ = module(
        tmp_path,
        monkeypatch,
        naturalistic_manifest=str(path),
        masked_crops_per_recording=8,
        masked_crop_samples=8,
        allow_stream_restart_on_resume=True,
    )
    iterator = iter(data.train_dataloader())
    batch = next(iterator)
    assert batch["waveform"].shape == (2, 2, 8)
    assert batch["masked_crops"].shape == (2, 8, 2, 8)
    assert torch.isfinite(batch["masked_crops"]).all()
    assert not torch.equal(batch["waveform"][:, 0], batch["waveform"][:, 1])
    del iterator
    state = data.state_dict()
    data.load_state_dict(state)
    assert data._stream_seed == state["stream_seed"] + 1
    for field in ("naturalistic_sha256", "masked_crops_per_recording", "masked_crop_samples"):
        damaged = deepcopy(state)
        damaged[field] = None
        with pytest.raises(ValueError, match="naturalistic assets or masked crop"):
            data._check_resume(damaged)
    # A later worker must not silently consume a modified asset declaration.
    manifest["provenance"]["revision"] = "changed-after-setup"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest changed after data setup"):
        next(iter(data.train_dataloader()))


@pytest.mark.parametrize("mismatch", ["bucket", "endpoint", "cache"])
def test_nat_remote_scope_and_per_consumer_capacity_checked_before_training(
    tmp_path, monkeypatch, mismatch
):
    path, manifest = _assets(tmp_path)
    _remote_assets(path, manifest, packed="npz")
    manifest.update(bucket="private-bucket", endpoint=ENDPOINT)
    kwargs = {}
    if mismatch == "bucket":
        manifest["bucket"] = "unrelated-bucket"
    elif mismatch == "endpoint":
        manifest["endpoint"] = "https://another.invalid"
    else:
        kwargs["naturalistic_cache_gib"] = 0.000001
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="AudioSet bucket and endpoint|bytes fitting the cache"):
        module(tmp_path, monkeypatch, naturalistic_manifest=str(path), **kwargs)


def test_silent_staged_pcm_nat_decode_and_backward_remain_finite(tmp_path):
    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    row = _decode(
        make_wav([0] * 2048),
        "silent-recording.wav",
        16000,
        1152,
        7,
        0,
        True,
        naturalistic=NaturalisticAudio(path, require_asset_checksums=True),
        crops_per_recording=8,
        crop_samples=256,
    )
    batch = {name: torch.stack([value, value]) for name, value in row.items()}
    assert batch["waveform"].shape == (2, 2, 1152)
    assert torch.count_nonzero(batch["waveform"]) == 0
    model = nat_model()
    loss = model.compute_loss(**batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

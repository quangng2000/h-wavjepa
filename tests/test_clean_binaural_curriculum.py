"""Clean-to-Nat data transitions retain one model and reject unrelated resumes."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from test_naturalistic_audio import _assets, _with_checksums
from test_sharded_audio import make_wav, module

from src.data.sharded_audio_datamodule import _decode, data_state_sha256
from src.utils.runtime import validate_config


def test_clean_binaural_duplicates_one_recording_before_both_crop_paths():
    kwargs = dict(
        raw=make_wav(range(100)),
        name="recording.wav",
        sample_rate=16000,
        num_samples=32,
        seed=17,
        cycle=2,
        random_crop=True,
        crops_per_recording=8,
        crop_samples=16,
    )
    mono = _decode(**kwargs)
    clean = _decode(**kwargs, clean_binaural=True)
    assert clean["waveform"].shape == (2, 32)
    assert clean["masked_crops"].shape == (8, 2, 16)
    for ear in range(2):
        torch.testing.assert_close(clean["waveform"][ear], mono["waveform"], rtol=0, atol=0)
        torch.testing.assert_close(
            clean["masked_crops"][:, ear], mono["masked_crops"], rtol=0, atol=0
        )
    assert not torch.equal(clean["masked_crops"][0], clean["masked_crops"][1])
    with pytest.raises(ValueError, match="cannot also render"):
        _decode(**kwargs, clean_binaural=True, naturalistic=object())


def test_clean_binaural_stream_and_same_phase_resume(tmp_path, monkeypatch):
    data, _ = module(
        tmp_path,
        monkeypatch,
        clean_binaural=True,
        masked_crops_per_recording=8,
        masked_crop_samples=8,
        allow_stream_restart_on_resume=True,
    )
    iterator = iter(data.train_dataloader())
    batch = next(iterator)
    assert batch["waveform"].shape == (2, 2, 8)
    assert batch["masked_crops"].shape == (2, 8, 2, 8)
    assert torch.equal(batch["waveform"][:, 0], batch["waveform"][:, 1])
    assert torch.equal(batch["masked_crops"][:, :, 0], batch["masked_crops"][:, :, 1])
    del iterator
    state = data.state_dict()
    assert state["data_mode"] == "clean_binaural"
    assert state["naturalistic_sha256"] is None
    data.load_state_dict(state)
    assert data._stream_seed == state["stream_seed"] + 1
    for changes in ({"data_mode": "mono"}, {"data_mode": "naturalistic"}):
        with pytest.raises(ValueError, match="naturalistic assets or masked crop"):
            data._check_resume(state | changes)


def _curriculum(tmp_path, monkeypatch):
    clean_path, nat_path = tmp_path / "clean", tmp_path / "nat"
    clean_path.mkdir()
    nat_path.mkdir()
    clean, _ = module(
        clean_path,
        monkeypatch,
        clean_binaural=True,
        masked_crops_per_recording=8,
        masked_crop_samples=8,
        allow_stream_restart_on_resume=True,
    )
    source = clean.state_dict()
    manifest_path, manifest = _assets(nat_path)
    _with_checksums(manifest_path, manifest)
    nat, _ = module(
        nat_path,
        monkeypatch,
        naturalistic_manifest=str(manifest_path),
        masked_crops_per_recording=8,
        masked_crop_samples=8,
        allow_stream_restart_on_resume=True,
    )
    transition = {
        "source_checkpoint_sha256": "a" * 64,
        "source_data_state_sha256": data_state_sha256(source),
        "index_sha256": source["index_sha256"],
        "target_manifest_sha256": nat.naturalistic_sha256,
    }
    return clean, source, nat, transition


def test_one_way_transition_requires_exact_source_and_frozen_target(tmp_path, monkeypatch):
    clean, source, nat, transition = _curriculum(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="naturalistic assets or masked crop"):
        nat.load_state_dict(source)
    nat.hparams.clean_to_nat_transition = transition
    nat.load_state_dict(source)
    assert nat.state_dict()["data_mode"] == "naturalistic"
    assert nat.state_dict()["naturalistic_sha256"] == transition["target_manifest_sha256"]
    assert nat._stream_seed == source["stream_seed"] + 1
    with pytest.raises(ValueError, match="naturalistic assets or masked crop"):
        clean.load_state_dict(nat.state_dict())
    # A transition permit cannot silently become a general resume bypass.
    with pytest.raises(ValueError, match="pinned clean-to-Nat"):
        nat.load_state_dict(nat.state_dict())
    nat.hparams.clean_to_nat_transition = None
    nat.load_state_dict(nat.state_dict())
    with pytest.raises(ValueError, match="naturalistic assets or masked crop"):
        nat.load_state_dict(nat.state_dict() | {"naturalistic_sha256": "b" * 64})


def test_transition_constructor_accepts_hydra_mapping_but_not_unbound_permit(tmp_path, monkeypatch):
    _, source, nat, transition = _curriculum(tmp_path, monkeypatch)
    config = OmegaConf.create(
        {
            "_target_": "src.data.sharded_audio_datamodule.ShardedAudioDataModule",
            **dict(nat.hparams),
            "clean_to_nat_transition": transition,
        }
    )
    restored = instantiate(config)
    restored.setup("fit")
    restored.load_state_dict(source)
    for bad in (
        transition | {"extra": "a" * 64},
        transition | {"source_checkpoint_sha256": "not-a-checksum"},
        {key: value for key, value in transition.items() if key != "source_data_state_sha256"},
    ):
        with pytest.raises(ValueError, match="pinned source and Nat target"):
            type(nat)(**(dict(nat.hparams) | {"clean_to_nat_transition": bad}))
    with pytest.raises(ValueError, match="pinned source and Nat target"):
        type(nat)(
            **(
                dict(nat.hparams)
                | {"clean_to_nat_transition": transition, "naturalistic_manifest": None}
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        {"data_mode": "mono"},
        {"naturalistic_sha256": "c" * 64},
        {"stream_seed": 54},
        {"masked_crop_samples": 7},
        {"masked_crops_per_recording": 4},
    ],
)
def test_transition_rejects_changed_original_data_state(tmp_path, monkeypatch, change):
    _, source, nat, transition = _curriculum(tmp_path, monkeypatch)
    nat.hparams.clean_to_nat_transition = transition
    with pytest.raises(ValueError, match="pinned clean-to-Nat"):
        nat.load_state_dict(source | change)


@pytest.mark.parametrize(
    "field", ["index_sha256", "target_manifest_sha256", "source_data_state_sha256"]
)
def test_transition_rejects_wrong_pinned_hash(tmp_path, monkeypatch, field):
    _, source, nat, transition = _curriculum(tmp_path, monkeypatch)
    nat.hparams.clean_to_nat_transition = transition | {field: "f" * 64}
    with pytest.raises(ValueError, match="pinned clean-to-Nat"):
        nat.load_state_dict(source)


def test_mono_legacy_checkpoint_without_data_mode_still_resumes(tmp_path, monkeypatch):
    data, _ = module(tmp_path, monkeypatch, allow_stream_restart_on_resume=True)
    state = data.state_dict()
    state.pop("data_mode")
    data.load_state_dict(state)


def test_clean_config_keeps_exact_nat_model_and_batch_recipe():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        clean = compose(
            config_name="train",
            overrides=[
                "experiment=audioset_clean_binaural_h100_four",
                "data.index_path=/explicit/index.json",
            ],
        )
        nat = compose(
            config_name="train",
            overrides=[
                "experiment=audioset_nat_h100_four",
                "data.index_path=/explicit/index.json",
                "data.naturalistic_manifest=/explicit/manifest.json",
            ],
        )
    validate_config(clean)
    validate_config(nat)
    assert OmegaConf.to_container(clean.model) == OmegaConf.to_container(nat.model)
    assert clean.data.naturalistic_manifest is None
    assert clean.data.clean_binaural is True
    assert clean.data.batch_size * clean.trainer.devices == 16
    assert clean.data.masked_crops_per_recording == 8
    assert clean.data.masked_crop_samples == 32160
    for changes in (
        {"naturalistic_manifest": "/also/nat.json"},
        {"clean_binaural": False},
        {"clean_binaural": "yes"},
    ):
        bad = deepcopy(clean)
        for name, value in changes.items():
            bad.data[name] = value
        with pytest.raises(ValueError, match="clean_binaural|Binaural staged"):
            validate_config(bad)

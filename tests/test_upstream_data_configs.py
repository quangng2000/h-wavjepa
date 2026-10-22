import json
import struct
import wave

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data.wav_datamodule import (
    WAVManifestDataModule,
    WAVManifestDataset,
    _EpochDistributedSampler,
    _MinimumBatchSampler,
)
from src.utils.runtime import validate_config


def _wav(path, frames):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack(f"<{frames}h", *range(frames)))
    return path


def test_forecasting_rejects_padding_but_legacy_keeps_it(tmp_path):
    audio = _wav(tmp_path / "short.wav", 8)
    legacy = WAVManifestDataset([audio], num_samples=16)
    torch.testing.assert_close(legacy[0]["waveform"][8:], torch.zeros(8))
    with pytest.raises(ValueError, match="padding cannot be used as future audio"):
        WAVManifestDataset([audio], num_samples=16, require_full_length=True)


@pytest.mark.parametrize("frames", [16, 20])
def test_forecasting_uses_contiguous_real_samples(tmp_path, frames):
    audio = _wav(tmp_path / "complete.wav", frames)
    dataset = WAVManifestDataset([audio], num_samples=16, require_full_length=True)
    torch.testing.assert_close(dataset[0]["waveform"], torch.arange(16) / 32768.0)


def test_forecasting_rechecks_length_when_loading(tmp_path):
    audio = _wav(tmp_path / "changed.wav", 16)
    dataset = WAVManifestDataset([audio], num_samples=16, require_full_length=True)
    _wav(audio, 8)
    with pytest.raises(ValueError, match="at least 16 real samples"):
        dataset[0]


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_full_length_option_must_be_boolean(tmp_path, flag):
    audio = _wav(tmp_path / "audio.wav", 16)
    with pytest.raises(ValueError, match="require_full_length must be a boolean"):
        WAVManifestDataset([audio], require_full_length=flag)
    with pytest.raises(ValueError, match="require_full_length must be a boolean"):
        WAVManifestDataModule("unused", "unused", "unused", require_full_length=flag)


def test_datamodule_propagates_full_length_requirement_to_every_split(tmp_path):
    manifests = {}
    for split in ("train", "val", "test"):
        audio = _wav(tmp_path / f"{split}.wav", 8 if split == "test" else 16)
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps({"path": audio.name}) + "\n", encoding="utf-8")
        manifests[f"{split}_manifest"] = str(manifest)
    module = WAVManifestDataModule(**manifests, num_samples=16, require_full_length=True)
    with pytest.raises(ValueError, match="test.wav"):
        module.setup()
    assert module.data_train is module.data_val is module.data_test is None


@pytest.mark.parametrize(
    ("size", "batch_size", "expected_sizes"),
    [
        (2, 2, [2]),
        (3, 2, [3]),
        (5, 2, [2, 3]),
        (6, 2, [2, 2, 2]),
        (7, 3, [3, 4]),
        (8, 3, [3, 3, 2]),
    ],
)
def test_minimum_batches_preserve_every_example(size, batch_size, expected_sizes):
    records = list(range(size))
    sampler = _MinimumBatchSampler(SequentialSampler(records), batch_size, minimum_batch_size=2)
    loader = DataLoader(records, batch_sampler=sampler)
    batches = list(loader)
    assert [len(batch) for batch in batches] == expected_sizes
    assert len(loader) == len(batches)
    assert torch.cat(batches).tolist() == records


def test_minimum_batches_preserve_shuffled_sampler_order():
    records = list(range(7))
    expected = list(RandomSampler(records, generator=torch.Generator().manual_seed(123)))
    sampler = _MinimumBatchSampler(
        RandomSampler(records, generator=torch.Generator().manual_seed(123)), 2, 2
    )
    batches = list(sampler)
    actual = [index for batch in batches for index in batch]
    assert actual == expected
    assert actual != records
    assert sorted(actual) == records
    assert [len(batch) for batch in batches] == [2, 2, 3]


def _split_manifests(tmp_path, size):
    manifests = {}
    for split in ("train", "val", "test"):
        rows = []
        for index in range(size):
            audio = _wav(tmp_path / f"{split}-{index}.wav", 16)
            rows.append({"path": audio.name})
        path = tmp_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        manifests[f"{split}_manifest"] = str(path)
    return manifests


def test_real_loader_merges_singleton_train_validation_and_test(tmp_path):
    manifests = _split_manifests(tmp_path, 5)
    module = WAVManifestDataModule(
        **manifests, num_samples=16, batch_size=2, require_full_length=True, minimum_batch_size=2
    )
    module.setup()
    for stage in ("train", "val", "test"):
        loader = getattr(module, f"{stage}_dataloader")()
        batch_sampler = loader.batch_sampler
        assert isinstance(batch_sampler, _MinimumBatchSampler)
        assert isinstance(batch_sampler.sampler, _EpochDistributedSampler)
        assert batch_sampler.sampler.shuffle == (stage == "train")
        batches = list(batch_sampler)
        assert sorted(index for batch in batches for index, epoch in batch) == list(range(5))
        assert all(epoch == 0 for batch in batches for index, epoch in batch)
        assert [batch["waveform"].shape[0] for batch in loader] == [2, 3]


def test_singleton_split_is_rejected_before_assignment(tmp_path):
    manifests = _split_manifests(tmp_path, 1)
    module = WAVManifestDataModule(**manifests, batch_size=2, minimum_batch_size=2)
    with pytest.raises(ValueError, match="train split has 1 recordings"):
        module.setup()
    assert module.data_train is module.data_val is module.data_test is None


@pytest.mark.parametrize("minimum", [0, True, 1.5, "2", 3])
def test_invalid_minimum_batch_size_rejected(minimum):
    with pytest.raises(ValueError, match="minimum_batch_size"):
        WAVManifestDataModule(
            "unused", "unused", "unused", batch_size=2, minimum_batch_size=minimum
        )


def test_legacy_loader_still_keeps_unmerged_singleton(tmp_path):
    manifests = _split_manifests(tmp_path, 3)
    module = WAVManifestDataModule(**manifests, num_samples=16, batch_size=2)
    module.setup()
    assert [batch["waveform"].shape[0] for batch in module.val_dataloader()] == [2, 1]


@pytest.mark.parametrize(
    ("experiment", "hierarchy", "weights", "regularization"),
    [
        ("upstream_hierarchical", True, [1.0, 1.0, 1.0], 0.1),
        ("upstream_flat", False, [1.0, 1.0, 1.0], 0.1),
        ("upstream_baseline", False, [0.0, 0.0, 0.0], 0.0),
    ],
)
def test_upstream_real_presets_are_explicit_and_chronological(
    make_config, experiment, hierarchy, weights, regularization
):
    cfg = make_config([f"experiment={experiment}"])
    validate_config(cfg)
    assert cfg.model._target_.endswith("hierarchical_wavjepa_module.HierarchicalWavJEPAModule")
    assert cfg.model.backbone._target_.endswith("wavjepa_backbone.WavJEPABackbone")
    assert cfg.model.backbone.checkpoint_path is None
    assert cfg.model.hierarchy is hierarchy
    assert list(cfg.model.prediction_weights) == weights
    assert cfg.model.regularization_weight == regularization
    assert cfg.data.require_full_length is True
    assert cfg.data.minimum_batch_size == 2
    assert cfg.data.num_samples == 32160 * (5 + 4)
    assert cfg.data.num_samples / cfg.data.sample_rate == 18.09
    assert OmegaConf.is_missing(cfg.data, "train_manifest")
    assert OmegaConf.is_missing(cfg.data, "val_manifest")
    assert OmegaConf.is_missing(cfg.data, "test_manifest")


@pytest.mark.parametrize(
    "experiment",
    [
        "audioset",
        "audioset_pretrained",
        "audioset_flat",
        "audioset_h100",
    ],
)
def test_audioset_presets_fit_real_ten_second_clips(make_config, experiment):
    cfg = make_config([f"experiment={experiment}"])
    validate_config(cfg)
    assert cfg.model.backbone.config.window_samples == 16160
    assert cfg.data.num_samples == 16160 * (5 + 4) == 145440
    assert cfg.data.num_samples / cfg.data.sample_rate == 9.09
    assert cfg.data.require_full_length
    assert cfg.data.train_random_crop and cfg.data.train_drop_last
    assert cfg.model.hierarchy == (experiment != "audioset_flat")
    assert cfg.model.backbone.pretrained == (experiment == "audioset_pretrained")
    assert cfg.model.lr_total_steps == cfg.trainer.max_steps == 375000
    assert cfg.model.lr_warmup_steps == cfg.model.ema_anneal_end_step == 100000
    assert cfg.model.readout_mode == "normalized_top_k"
    assert cfg.callbacks.early_stopping is None
    assert cfg.callbacks.model_checkpoint.filename == "step_{step:07d}"
    assert cfg.callbacks.model_checkpoint.save_on_train_epoch_end is False
    if experiment == "audioset_h100":
        assert cfg.trainer.devices == 2
        assert cfg.trainer.accelerator == "gpu"
        assert cfg.trainer.precision == "bf16-mixed"
        assert not cfg.trainer.use_distributed_sampler
        assert not cfg.trainer.deterministic
        assert cfg.model.backbone.config.use_gradient_checkpointing
        assert cfg.data.batch_size * cfg.trainer.devices == 64
    else:
        assert cfg.trainer.devices == 1
        assert cfg.trainer.precision == "32-true"


@pytest.mark.parametrize("name", ["train", "eval"])
def test_upstream_smoke_composes_without_downloads(make_config, name):
    cfg = make_config(["experiment=upstream_smoke"], name=name)
    validate_config(cfg)
    assert cfg.data.num_samples == 512 * 9
    assert cfg.model.backbone.config.encoder_dim == 16
    assert cfg.model.backbone.config.encoder_layers == 1
    assert cfg.model.backbone.config.average_top_k_layers == 1
    assert cfg.model.backbone.checkpoint_path is None
    assert cfg.trainer.accelerator == "cpu"
    assert "not-a-benchmark" in cfg.tags


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("data.num_samples=4607", "data.num_samples must equal"),
        ("data.sample_rate=8000", "data.sample_rate must match"),
        ("model.context_windows=0", "context_windows must be a positive integer"),
        ("model.context_windows=6", "context_windows - 1 must be divisible"),
        ("model.horizons=[1,2]", "requires three horizons"),
        ("model.horizons=[1,2,2]", "horizons must be strictly increasing"),
        ("model.horizons=[1,2,0]", "horizons must contain positive integers"),
        ("model.state_strides=[2,2,4]", "state_strides must start at 1"),
        ("model.state_strides=[1,4,2]", "state_strides must start at 1"),
        ("trainer.precision=16-mixed", "FP32 or bf16-mixed"),
        ("trainer.accelerator=mps", "MPS seeded RNG isolation"),
        ("trainer.accelerator=auto", "explicit cpu/cuda/gpu"),
        ("data.batch_size=1", "regularization requires data.batch_size >= 2"),
        ("trainer.devices=auto", "explicit positive devices"),
        ("trainer.devices=[0,0]", "explicit positive devices"),
        ("+trainer.num_nodes=0", "explicit positive devices"),
        ("+trainer.strategy=fsdp", "DDP strategies only"),
    ],
)
def test_upstream_preflight_rejects_unsupported_input_and_runtime(make_config, override, message):
    cfg = make_config(["experiment=upstream_smoke", override])
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_upstream_real_audio_preflight_requires_no_padding(make_config):
    cfg = make_config(["experiment=upstream_hierarchical", "data.require_full_length=false"])
    with pytest.raises(ValueError, match="require_full_length=true"):
        validate_config(cfg)


def test_upstream_real_audio_preflight_requires_minimum_batch_size(make_config):
    cfg = make_config(["experiment=upstream_hierarchical", "data.minimum_batch_size=1"])
    with pytest.raises(ValueError, match="regularization requires data.minimum_batch_size >= 2"):
        validate_config(cfg)


def test_no_regularization_allows_batch_size_one(make_config):
    cfg = make_config(
        ["experiment=upstream_smoke", "model.regularization_weight=0", "data.batch_size=1"]
    )
    validate_config(cfg)


def test_explicit_single_device_list_is_supported(make_config):
    cfg = make_config(["experiment=upstream_smoke", "trainer.devices=[0]"])
    validate_config(cfg)


@pytest.mark.parametrize(
    "overrides",
    [
        ["trainer.devices=2", "+trainer.strategy=ddp"],
        ["trainer.devices=[0,1]", "trainer.accelerator=gpu", "trainer.precision=bf16-mixed"],
        ["+trainer.num_nodes=2", "+trainer.strategy=ddp"],
    ],
)
def test_explicit_distributed_and_mixed_precision_configs_are_supported(make_config, overrides):
    cfg = make_config(["experiment=upstream_smoke", *overrides])
    validate_config(cfg)


def test_wav_distributed_config_requires_datamodule_sampler(make_config):
    cfg = make_config(
        [
            "experiment=upstream_hierarchical",
            "trainer.devices=2",
            "++trainer.use_distributed_sampler=true",
        ]
    )
    with pytest.raises(ValueError, match="use_distributed_sampler=false"):
        validate_config(cfg)
    cfg.trainer.use_distributed_sampler = False
    validate_config(cfg)


def test_legacy_smoke_still_uses_original_model(make_config):
    cfg = make_config(["experiment=smoke"])
    validate_config(cfg)
    assert cfg.model._target_ == "src.models.hwavjepa_module.HWavJEPAModule"
    assert cfg.data.num_samples == 1024

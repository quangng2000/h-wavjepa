"""Tiny analytic assets exercise Nat rendering without downloading any datasets."""

import hashlib
import io
import json
import wave

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.naturalistic_audio import (
    NaturalisticAudio,
    convolve_with_rir,
    fade_noise,
    generator_for_clip,
)


def _wav(path, samples, sample_rate=16000):
    data = np.asarray(samples)
    if data.ndim == 1:
        data = data[None, :]
    pcm = (data.clip(-1, 1) * 30000).astype("<i2").T.copy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(data.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def _assets(tmp_path, noise_frames=8000):
    source = np.array([[1, 0, 0, 0.25], [0.25, 0, 0, 0.0625]], dtype=np.float32)
    noise = np.array([[0.125, 0, 0.125], [1, 0, 1]], dtype=np.float32)
    np.save(tmp_path / "source.npy", source)
    for index in range(3):
        np.save(tmp_path / f"noise{index}.npy", noise * (index + 1))
    _wav(tmp_path / "whamr.wav", np.sin(np.arange(noise_frames) * 0.11))
    manifest = {
        "schema_version": 1,
        "sample_rate": 16000,
        "split": "train",
        "provenance": {"brir_dataset": "synthetic unit-test fixtures"},
        "scenes": [
            {
                "id": "localized-room",
                "noise_type": "localized",
                "source_rir": "source.npy",
                "noise_rirs": ["noise0.npy"],
            },
            {
                "id": "diffuse-room",
                "noise_type": "diffuse",
                "source_rir": "source.npy",
                "noise_rirs": [f"noise{i}.npy" for i in range(3)],
            },
        ],
        "noise_files": ["whamr.wav"],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def _source(length=16000):
    return torch.sin(torch.arange(length) * 0.031) + 0.1 * torch.cos(torch.arange(length) * 0.07)


def test_fft_renderer_matches_direct_causal_convolution_and_keeps_ear_gain():
    waveform = torch.tensor([1.0, 2.0, -1.0, 0.0, 3.0])
    rir = torch.tensor([[0.5, 0.0, 1.0], [0.0, 0.2, 0.0]])
    expected = F.conv1d(F.pad(waveform[None, None], (2, 2)), rir.flip(-1)[:, None])[0]
    actual = convolve_with_rir(waveform, rir)
    torch.testing.assert_close(actual, expected[:, : waveform.numel()], atol=1e-6, rtol=1e-6)


def test_render_full_recording_then_crop_retains_pre_crop_reverberation():
    waveform = torch.zeros(32)
    waveform[7] = 1.0
    rir = torch.zeros(2, 6)
    rir[:, 5] = torch.tensor([1.0, 0.5])
    full_then_crop = convolve_with_rir(waveform, rir)[:, 10:20]
    crop_first = convolve_with_rir(waveform[10:20], rir)
    torch.testing.assert_close(full_then_crop[:, 2], torch.tensor([1.0, 0.5]))
    assert torch.count_nonzero(crop_first) == 0


def test_manifest_has_stable_identity_and_explicit_provenance(tmp_path):
    path, _ = _assets(tmp_path)
    augmenter = NaturalisticAudio(path)
    assert augmenter.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert augmenter.split == "train"
    assert augmenter.clean_probability == 0
    assert augmenter.provenance["brir_dataset"] == "synthetic unit-test fixtures"


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "diffuse_probability,kind,count", [(0.0, "localized", 1), (1.0, "diffuse", 3)]
)
def test_rendered_scene_snr_and_binaural_levels(
    tmp_path, normalize, diffuse_probability, kind, count
):
    path, _ = _assets(tmp_path)
    augmenter = NaturalisticAudio(
        path, snr_db=(17.0, 17.0), diffuse_probability=diffuse_probability, normalize=normalize
    )
    result = augmenter.render(_source(), generator=generator_for_clip(42, "clip-id", 3))
    assert result.waveform.shape == (2, 16000)
    assert torch.isfinite(result.waveform).all()
    assert result.metadata["noise_type"] == kind
    assert result.metadata["noise_source_count"] == count
    assert result.metadata["snr_db"] == 17
    measured = 10 * torch.log10(
        result.source.double().square().sum() / result.noise.double().square().sum()
    )
    assert measured.item() == pytest.approx(17, abs=1e-5)
    # Joint SNR and normalization retain the distinct source and noise ILDs.
    torch.testing.assert_close(result.source[0], 4 * result.source[1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.noise[1], 8 * result.noise[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result.waveform, result.source + result.noise)
    if normalize:
        assert result.waveform.square().mean().sqrt().item() == pytest.approx(1, abs=1e-6)


def test_clip_epoch_generator_is_reproducible_and_does_not_consume_global_rng(tmp_path):
    path, _ = _assets(tmp_path)
    augmenter = NaturalisticAudio(path)
    before = torch.random.get_rng_state().clone()
    first = augmenter(_source(), generator=generator_for_clip(41, "clip-a", 2))
    second = augmenter(_source(), generator=generator_for_clip(41, "clip-a", 2))
    changed_epoch = augmenter(_source(), generator=generator_for_clip(41, "clip-a", 3))
    changed_clip = augmenter(_source(), generator=generator_for_clip(41, "clip-b", 2))
    assert torch.equal(first, second)
    assert not torch.equal(first, changed_epoch)
    assert not torch.equal(first, changed_clip)
    assert torch.equal(before, torch.random.get_rng_state())


def test_half_localized_half_diffuse_and_snr_sampling(tmp_path):
    path, _ = _assets(tmp_path, noise_frames=128)
    augmenter = NaturalisticAudio(path)
    counts = {"localized": 0, "diffuse": 0}
    values = []
    for index in range(400):
        result = augmenter.render(_source(256), generator=generator_for_clip(42, str(index), 0))
        counts[result.metadata["noise_type"]] += 1
        values.append(result.metadata["snr_db"])
    assert 160 <= counts["localized"] <= 240
    assert 160 <= counts["diffuse"] <= 240
    assert all(5 <= value <= 40 for value in values)
    assert min(values) < 7 and max(values) > 38


def test_noise_has_200ms_fades_without_mutating_source():
    waveform = torch.ones(16000)
    faded = fade_noise(waveform)
    assert faded[0] == faded[-1] == 0
    assert faded[3199] == faded[-3200] == 1
    assert torch.all(faded[3200:-3200] == 1)
    assert torch.all(waveform == 1)
    assert fade_noise(torch.ones(128)).shape == (128,)


def test_short_noise_tail_survives_past_dry_noise_end(tmp_path):
    path, _ = _assets(tmp_path, noise_frames=64)
    rir = np.zeros((2, 17), dtype=np.float32)
    rir[:, -1] = 1
    np.save(tmp_path / "noise0.npy", rir)
    augmenter = NaturalisticAudio(path, diffuse_probability=0, normalize=False)
    result = augmenter.render(_source(256), generator=generator_for_clip(42, "tail", 0))
    start, length = result.metadata["noise_start_samples"], result.metadata["noise_samples"]
    assert start + length + 16 < 256  # This deterministic draw keeps the tail in bounds.
    assert result.noise[:, start + length : start + length + 15].abs().max() > 0.001


def test_noise_is_trimmed_to_ten_seconds_and_source_duration_is_bounded(tmp_path):
    path, _ = _assets(tmp_path, noise_frames=160_001)
    augmenter = NaturalisticAudio(path, diffuse_probability=0)
    result = augmenter.render(_source(160_001), generator=generator_for_clip(4, "trim", 0))
    assert result.metadata["noise_samples"] == 160_000
    with pytest.raises(ValueError, match="duration bound"):
        augmenter(torch.zeros(480_001), generator=generator_for_clip(1, "too-long", 0))


def test_explicit_clean_ablation_is_binaural_but_missing_assets_never_fallback(tmp_path):
    path, _ = _assets(tmp_path)
    augmenter = NaturalisticAudio(path, clean_probability=1)
    result = augmenter.render(_source(512), generator=generator_for_clip(4, "clean", 0))
    assert result.metadata["clean"]
    torch.testing.assert_close(result.waveform[0], result.waveform[1])
    assert torch.count_nonzero(result.noise) == 0
    (tmp_path / "noise0.npy").unlink()
    with pytest.raises(ValueError, match="missing"):
        NaturalisticAudio(path, clean_probability=1)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("sample_rate", 32000, "sample_rate"),
        ("schema_version", 2, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("split", "unspecified", "split"),
        ("scenes", [], "nonempty scenes"),
        ("noise_files", [], "noise_files"),
        ("provenance", "unknown", "provenance"),
    ],
)
def test_manifest_validation(tmp_path, field, value, message):
    path, manifest = _assets(tmp_path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        NaturalisticAudio(path)


@pytest.mark.parametrize("count", [0, 1, 2, 6])
def test_diffuse_requires_three_to_five_distinct_scene_noise_brirs(tmp_path, count):
    path, manifest = _assets(tmp_path)
    manifest["scenes"][1]["noise_rirs"] = ["noise0.npy"] * count
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="require 3-5"):
        NaturalisticAudio(path)


def test_diffuse_rejects_repeating_the_same_location_as_fake_diffusion(tmp_path):
    path, manifest = _assets(tmp_path)
    manifest["scenes"][1]["noise_rirs"] = ["noise0.npy"] * 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="distinct"):
        NaturalisticAudio(path)


def test_missing_scene_population_does_not_change_requested_mix(tmp_path):
    path, manifest = _assets(tmp_path)
    manifest["scenes"] = manifest["scenes"][:1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no diffuse"):
        NaturalisticAudio(path)
    assert len(NaturalisticAudio(path, diffuse_probability=0).scenes["localized"]) == 1


@pytest.mark.parametrize("asset", ["noise", "rir"])
def test_asset_sample_rate_mismatch_is_rejected(tmp_path, asset):
    path, manifest = _assets(tmp_path)
    if asset == "noise":
        _wav(tmp_path / "whamr.wav", np.sin(np.arange(8000)), sample_rate=32000)
    else:
        _wav(tmp_path / "source.wav", np.ones((2, 16)) * 0.1, sample_rate=32000)
        manifest["scenes"][0]["source_rir"] = "source.wav"
        path.write_text(json.dumps(manifest))
    augmenter = NaturalisticAudio(path, diffuse_probability=0)
    with pytest.raises(ValueError, match="sample rate"):
        augmenter(_source(), generator=generator_for_clip(1, "mismatch", 0))


@pytest.mark.parametrize(
    "value", [np.zeros((2, 4)), np.ones((1, 4)), np.ones((2, 48001)), np.full((2, 4), np.nan)]
)
def test_invalid_brirs_are_rejected_without_dry_fallback(tmp_path, value):
    path, _ = _assets(tmp_path)
    np.save(tmp_path / "source.npy", value)
    augmenter = NaturalisticAudio(path, diffuse_probability=0)
    with pytest.raises(ValueError, match="BRIR"):
        augmenter(_source(), generator=generator_for_clip(1, "bad-rir", 0))


def test_stereo_noise_is_not_silently_flattened_or_downmixed(tmp_path):
    path, _ = _assets(tmp_path)
    _wav(tmp_path / "whamr.wav", np.ones((2, 128)) * 0.2)
    augmenter = NaturalisticAudio(path)
    with pytest.raises(ValueError, match="1-channel"):
        augmenter(_source(), generator=generator_for_clip(1, "stereo-noise", 0))


def test_nonfinite_source_and_silent_noise_fail_explicitly(tmp_path):
    path, _ = _assets(tmp_path)
    augmenter = NaturalisticAudio(path)
    with pytest.raises(ValueError, match="finite"):
        augmenter(torch.full((128,), float("nan")), generator=generator_for_clip(1, "invalid", 0))
    _wav(tmp_path / "whamr.wav", np.zeros(128))
    with pytest.raises(ValueError, match="positive energy"):
        augmenter(_source(), generator=generator_for_clip(1, "silent-noise", 0))


@pytest.mark.parametrize("source", [torch.ones(128), torch.zeros(128)])
def test_degenerate_source_keeps_sample_and_records_undefined_snr(tmp_path, source):
    path, _ = _assets(tmp_path)
    result = NaturalisticAudio(path).render(source, generator=generator_for_clip(3, "silent", 0))
    assert torch.equal(result.waveform, torch.zeros(2, 128))
    assert torch.equal(result.source, result.noise)
    assert result.metadata["source_degenerate"] is True
    assert result.metadata["snr_db"] is None
    assert result.metadata["noise_scale"] == 0
    assert 5 <= result.metadata["requested_snr_db"] <= 40


def _with_checksums(path, manifest):
    paths = {"source.npy", "noise0.npy", "noise1.npy", "noise2.npy", "whamr.wav"}
    manifest["asset_sha256"] = {
        name: hashlib.sha256((path.parent / name).read_bytes()).hexdigest() for name in paths
    }
    path.write_text(json.dumps(manifest))


def test_production_requires_complete_asset_checksums(tmp_path):
    path, manifest = _assets(tmp_path)
    with pytest.raises(ValueError, match="complete asset_sha256"):
        NaturalisticAudio(path, require_asset_checksums=True)
    _with_checksums(path, manifest)
    augmenter = NaturalisticAudio(path, require_asset_checksums=True)
    assert not augmenter._verified_assets  # Initialization does not hash the whole corpus.
    result = augmenter(_source(), generator=generator_for_clip(1, "verified", 0))
    assert torch.isfinite(result).all()
    assert augmenter._verified_assets
    manifest["asset_sha256"].pop("source.npy")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover exactly"):
        NaturalisticAudio(path, require_asset_checksums=True)


@pytest.mark.parametrize("mutated", ["source.npy", "whamr.wav"])
def test_asset_hash_mismatch_after_initial_use_cannot_hide_in_cache(tmp_path, mutated):
    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    augmenter = NaturalisticAudio(path, require_asset_checksums=True)
    augmenter(_source(), generator=generator_for_clip(1, "verified", 0))
    target = tmp_path / mutated
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        augmenter(_source(), generator=generator_for_clip(1, "verified", 0))


def test_unchanged_asset_is_not_rehashed_on_every_clip(tmp_path, monkeypatch):
    path, manifest = _assets(tmp_path)
    _with_checksums(path, manifest)
    augmenter = NaturalisticAudio(path, diffuse_probability=0, require_asset_checksums=True)
    augmenter(_source(), generator=generator_for_clip(1, "verified", 0))
    original_open = type(path).open

    def guard_open(asset, *args, **kwargs):
        if asset in augmenter._verified_assets:
            raise AssertionError("unchanged cached asset was rehashed")
        return original_open(asset, *args, **kwargs)

    monkeypatch.setattr(type(path), "open", guard_open)
    # numpy and wave use builtin open; the explicit checksum path uses Path.open.
    augmenter(_source(), generator=generator_for_clip(1, "verified", 0))


class _FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def get_object(self, *, Bucket, Key):
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key])}


def _remote_assets(path, manifest, *, packed=None):
    names = ["source.npy", "noise0.npy", "noise1.npy", "noise2.npy", "whamr.wav"]
    objects, assets = {}, {}
    for index, name in enumerate(names):
        if packed and name.endswith(".npy"):
            key = f"nat/packed-scene.{packed}"
            if key not in objects:
                arrays = np.stack(
                    [
                        np.load(path.parent / item)
                        if item == "source.npy"
                        else F.pad(torch.from_numpy(np.load(path.parent / item)), (0, 1)).numpy()
                        for item in names[:-1]
                    ]
                )
                buffer = io.BytesIO()
                if packed == "npy":
                    np.save(buffer, arrays)
                else:
                    np.savez_compressed(buffer, arrays)
                objects[key] = buffer.getvalue()
            payload = objects[key]
        else:
            key = f"nat/{name}"
            payload = (path.parent / name).read_bytes()
            objects[key] = payload
        assets[name] = {
            "key": key,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if packed and name.endswith(".npy"):
            assets[name]["index"] = index
    manifest.update(
        assets=assets, bucket="private-audioset", endpoint="https://storage.givemeanode.com"
    )
    # The logical extension is the decoder contract, not the cache's .tar suffix.
    if packed == "npz":
        assets_npz = {}
        for name, asset in assets.items():
            assets_npz[name.replace(".npy", ".npz")] = asset
        manifest["assets"] = assets_npz
        for scene in manifest["scenes"]:
            scene["source_rir"] = scene["source_rir"].replace(".npy", ".npz")
            scene["noise_rirs"] = [name.replace(".npy", ".npz") for name in scene["noise_rirs"]]
    path.write_text(json.dumps(manifest))
    return _FakeS3(objects)


@pytest.mark.parametrize("packed", [None, "npy", "npz"])
def test_private_s3_assets_render_with_bounded_verified_cache(tmp_path, monkeypatch, packed):
    from src.data import sharded_audio_datamodule as streaming

    path, manifest = _assets(tmp_path)
    local = NaturalisticAudio(path, diffuse_probability=1)
    expected = local(_source(), generator=generator_for_clip(3, "same", 0))
    store = _remote_assets(path, manifest, packed=packed)
    # Remote construction neither requires the local assets nor contacts S3.
    for item in tmp_path.glob("*.npy"):
        item.unlink()
    (tmp_path / "whamr.wav").unlink()
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    budget = max(len(payload) for payload in store.objects.values())
    cache = tmp_path / "nat-cache"
    remote = NaturalisticAudio(
        path,
        diffuse_probability=1,
        require_asset_checksums=True,
        cache_dir=cache,
        cache_max_gib=(budget + 1) / 1024**3,
    )
    assert not store.calls and not cache.exists()
    actual = remote(_source(), generator=generator_for_clip(3, "same", 0))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert sum(item.stat().st_size for item in cache.glob("*.tar")) <= budget
    assert not list(cache.glob("*.partial"))
    assert {bucket for bucket, _ in store.calls} == {"private-audioset"}
    if packed:
        assert len(store.calls) == 2  # One WHAMR file and one scene shared by all BRIR slices.
    else:
        assert len(store.calls) == 5


def test_remote_asset_corruption_fails_and_preserves_no_partial_object(tmp_path, monkeypatch):
    from src.data import sharded_audio_datamodule as streaming

    path, manifest = _assets(tmp_path)
    store = _remote_assets(path, manifest)
    store.objects["nat/whamr.wav"] = b"bad" + store.objects["nat/whamr.wav"][3:]
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    cache = tmp_path / "cache"
    augmenter = NaturalisticAudio(path, cache_dir=cache)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        augmenter(_source(), generator=generator_for_clip(3, "bad", 0))
    assert not list(cache.glob("*.tar")) and not list(cache.glob("*.partial"))


def test_remote_cache_revalidates_corrupted_cached_asset(tmp_path, monkeypatch):
    from src.data import sharded_audio_datamodule as streaming

    path, manifest = _assets(tmp_path)
    store = _remote_assets(path, manifest, packed="npy")
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    cache = tmp_path / "cache"
    augmenter = NaturalisticAudio(path, cache_dir=cache, diffuse_probability=1)
    first = augmenter(_source(), generator=generator_for_clip(3, "same", 0))
    packed_sha = manifest["assets"]["source.npy"]["sha256"]
    (cache / f"{packed_sha}.tar").write_bytes(b"bad")
    second = augmenter(_source(), generator=generator_for_clip(3, "same", 0))
    torch.testing.assert_close(first, second)
    assert [key for _, key in store.calls].count("nat/packed-scene.npy") == 2


def test_remote_validation_binds_object_size_hash_and_cache_contract(tmp_path, monkeypatch):
    from src.data import sharded_audio_datamodule as streaming

    path, manifest = _assets(tmp_path)
    store = _remote_assets(path, manifest)
    monkeypatch.setattr(streaming, "_s3_client", lambda endpoint: store)
    with pytest.raises(ValueError, match="bytes fitting the cache"):
        NaturalisticAudio(path, cache_max_gib=1 / 1024**3)
    augmenter = NaturalisticAudio(path)
    with pytest.raises(ValueError, match="explicit per-consumer cache_dir"):
        augmenter(_source(), generator=generator_for_clip(3, "missing-cache", 0))
    cache = tmp_path / "cache"
    augmenter = NaturalisticAudio(path, cache_dir=cache)
    augmenter(_source(), generator=generator_for_clip(3, "contract", 0))
    manifest["provenance"]["revision"] = "changed"
    path.write_text(json.dumps(manifest))
    changed = NaturalisticAudio(path, cache_dir=cache)
    with pytest.raises(ValueError, match="different asset contract"):
        changed(_source(), generator=generator_for_clip(3, "contract", 0))


def test_remote_duplicate_packed_slice_is_not_fake_diffuse_noise(tmp_path):
    path, manifest = _assets(tmp_path)
    _remote_assets(path, manifest, packed="npy")
    manifest["assets"]["noise2.npy"]["index"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="distinct remote locations"):
        NaturalisticAudio(path)


def test_full_three_second_rir_is_preserved(tmp_path):
    path, _ = _assets(tmp_path)
    rir = np.zeros((2, 48000), dtype=np.float32)
    rir[:, -1] = [1, 0.5]
    np.save(tmp_path / "source.npy", rir)
    augmenter = NaturalisticAudio(path, normalize=False)
    source = torch.zeros(64000)
    source[1] = 1
    reverberant = convolve_with_rir(source, augmenter._load_rir(tmp_path / "source.npy"))
    torch.testing.assert_close(reverberant[:, 48000], torch.tensor([1.0, 0.5]))
    assert reverberant[:, :32000].abs().max() < 1e-6


@pytest.mark.parametrize(
    "changes",
    [
        {"key": "../escape.npy"},
        {"sha256": "bad"},
        {"bytes": 0},
        {"index": True},
        {"index": 6},
    ],
)
def test_remote_invalid_object_metadata_fails_before_network(tmp_path, changes):
    path, manifest = _assets(tmp_path)
    _remote_assets(path, manifest, packed="npy")
    manifest["assets"]["source.npy"].update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Remote|Packed"):
        NaturalisticAudio(path)

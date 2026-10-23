"""Small synthetic tests only: no released weights, network, or GPU required."""

import array
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import probe_readout as probe
from src.pretrained import file_sha256


def summarize(states, batch_size):
    moments = probe.CohortMoments()
    for part in states.split(batch_size):
        moments.update(part)
    return moments.report()


def test_cohort_between_variance_is_not_mean_batch_between_variance():
    # Every batch has zero between variance, but the whole cohort does not.
    states = torch.tensor([0, 0, 4, 4], dtype=torch.float64)[:, None, None].expand(4, 5, 3)
    result = summarize(states, 2)
    assert result["between_recording_variance"] == 4
    assert result["within_recording_variance"] == 0
    assert result["within_variance_fraction"] == 0
    for batch_size in (1, 3, 4):
        assert summarize(states, batch_size) == result


@pytest.mark.parametrize("batch_size", [1, 2, 4, 11])
def test_streaming_moments_match_direct_cohort_statistics(batch_size):
    generator = torch.Generator().manual_seed(17)
    states = torch.randn(11, 9, 7, generator=generator, dtype=torch.float64)
    states += torch.arange(11, dtype=torch.float64)[:, None, None] * 20
    result = summarize(states, batch_size)
    expected_within = states.var(dim=1, correction=0).mean().item()
    expected_between = states.mean(dim=1).var(dim=0, correction=0).mean().item()
    assert result["within_recording_variance"] == pytest.approx(expected_within, rel=1e-12)
    assert result["between_recording_variance"] == pytest.approx(expected_between, rel=1e-12)
    assert result["total_variance"] == pytest.approx(
        states.flatten(0, 1).var(dim=0, correction=0).mean().item(), rel=1e-12
    )


def test_recording_constant_dynamic_and_fully_collapsed_have_distinct_reports():
    constants = torch.arange(4).double()[:, None, None].expand(4, 5, 3)
    dynamic = torch.arange(5).double()[None, :, None].expand(4, 5, 3)
    constant_result = summarize(constants, 2)
    dynamic_result = summarize(dynamic, 2)
    zero_result = summarize(torch.zeros_like(dynamic), 2)
    assert constant_result["between_recording_variance"] > 0
    assert constant_result["within_variance_fraction"] == 0
    assert constant_result["adjacent_window_change_mse"] == 0
    assert dynamic_result["between_recording_variance"] == 0
    assert dynamic_result["within_variance_fraction"] == 1
    assert dynamic_result["adjacent_window_change_mse"] == 1
    assert zero_result["within_recording_variance"] == 0
    assert zero_result["between_recording_variance"] == 0
    assert zero_result["within_variance_fraction"] == 0


@pytest.mark.parametrize(
    "states", [torch.zeros(2, 1, 3), torch.zeros(2, 3), torch.full((2, 3, 4), float("nan"))]
)
def test_invalid_states_rejected(states):
    with pytest.raises(ValueError):
        probe.CohortMoments().update(states)


def test_moments_require_two_recordings_and_consistent_geometry():
    moments = probe.CohortMoments()
    with pytest.raises(ValueError, match="two recordings"):
        moments.report()
    moments.update(torch.zeros(1, 3, 4))
    with pytest.raises(ValueError, match="two recordings"):
        moments.report()
    with pytest.raises(ValueError, match="same window count"):
        moments.update(torch.zeros(1, 4, 4))


def test_seeded_subset_is_distinct_and_reproducible():
    entries = [SimpleNamespace(identity=(1, i), recording_id=f"recording-{i}") for i in range(20)]
    entries += [SimpleNamespace(identity=(1, 0), recording_id="alias")]
    entries += [SimpleNamespace(identity=(1, 100), recording_id="recording-1")]
    first = probe.select_recordings(entries, 8, 31)
    assert first == probe.select_recordings(entries, 8, 31)
    assert first != probe.select_recordings(entries, 8, 32)
    assert len(first) == len(set(first)) == 8
    assert first == sorted(first)
    assert max(first) < 20
    assert probe.select_recordings(entries, 30, 31) == list(range(20))


def test_recording_aliases_are_resolved_before_subset_selection():
    entries = [
        SimpleNamespace(identity=(1, 0), recording_id="first"),
        SimpleNamespace(identity=(1, 1), recording_id="second"),
        SimpleNamespace(identity=(1, 1), recording_id="first"),
        SimpleNamespace(identity=(1, 2), recording_id="third"),
    ]
    assert probe.select_recordings(entries, 10, 0) == [0, 3]


def write_wav(path, samples):
    data = array.array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(data.tobytes())


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    weights = tmp_path / "synthetic-weights.bin"
    weights.write_bytes(b"Explicit synthetic test fixture, not released weights")
    sha = file_sha256(weights)
    # The actual command checks a fixed, pinned release digest. Only unit tests
    # replace it and the model so no real 785 MB weights need to be downloaded.
    monkeypatch.setattr(probe, "OFFICIAL_SHA256", sha)
    records = []
    for i in range(6):
        path = tmp_path / f"recording-{i}.wav"
        # Two leading samples prove the manifest offset is honored.
        samples = [30000, 30000] + [100 * i + t * 10 + k for t in range(9) for k in range(4)]
        write_wav(path, samples)
        records.append({"path": path.name, "recording_id": f"r-{i}", "offset_samples": 2})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    constructed = []

    class FakeBackbone(torch.nn.Module):
        def __init__(self, *, config, pretrained_path, local_files_only):
            super().__init__()
            assert local_files_only is True
            assert config == {"window_samples": 4}
            assert Path(pretrained_path) == weights
            self.checkpoint_sha256 = sha
            self.parameter = torch.nn.Parameter(torch.ones(()))
            self.calls = []
            constructed.append(self)

        def encode_windows(self, waveform, readout_mode):
            assert not self.training and not self.parameter.requires_grad
            assert torch.is_inference_mode_enabled()
            assert readout_mode == "normalized_top_k"
            self.calls.append(waveform.shape[0])
            return torch.stack((waveform.mean(-1), waveform.amax(-1)), dim=-1)

    monkeypatch.setattr(probe, "WavJEPABackbone", FakeBackbone)
    return {"weights": weights, "manifest": manifest, "window_samples": 4}, constructed


def test_bounded_probe_preserves_offsets_and_both_cohort_reports(local_inputs, tmp_path):
    inputs, constructed = local_inputs
    output = tmp_path / "report.json"
    result = probe.run_probe(
        **inputs, output=output, max_recordings=4, seed=37, batch_size=3, window_batch_size=5
    )
    assert json.loads(output.read_text()) == result
    assert result["status"] == "complete"
    assert result["cohort"]["manifest_sha256"] == file_sha256(inputs["manifest"])
    assert result["initialization"]["weights_sha256"] == file_sha256(inputs["weights"])
    assert result["initialization"]["official_sha256_verified"] is True
    assert result["statistics"]["recordings"] == 4
    assert result["statistics"]["windows_per_recording"] == 9
    assert result["observed_prefix_statistics"]["windows_per_recording"] == 5
    assert max(constructed[0].calls) <= 5
    assert sum(constructed[0].calls) == 4 * 9  # Prefix reuses the same readouts.
    samples = result["cohort"]["samples"]
    expected = (
        torch.tensor(
            [
                [
                    [
                        100 * sample["manifest_entry_index"] + t * 10 + 1.5,
                        100 * sample["manifest_entry_index"] + t * 10 + 3,
                    ]
                    for t in range(9)
                ]
                for sample in samples
            ],
            dtype=torch.float64,
        )
        / 32768
    )
    assert all(sample["offset_samples"] == 2 for sample in samples)
    for actual, expected_stats in (
        (result["statistics"], summarize(expected, 1)),
        (result["observed_prefix_statistics"], summarize(expected[:, :5], 1)),
    ):
        for key in (
            "within_recording_variance",
            "between_recording_variance",
            "mean_window_l2_norm",
        ):
            assert actual[key] == pytest.approx(expected_stats[key])


def test_probe_recording_and_window_batch_partition_invariance(local_inputs, tmp_path):
    inputs, _ = local_inputs
    first = probe.run_probe(
        **inputs, output=tmp_path / "first.json", batch_size=2, window_batch_size=3
    )
    second = probe.run_probe(
        **inputs, output=tmp_path / "second.json", batch_size=5, window_batch_size=17
    )
    assert first["cohort"] == second["cohort"]
    for section in ("statistics", "observed_prefix_statistics"):
        for key, value in first[section].items():
            if isinstance(value, float):
                assert second[section][key] == pytest.approx(value, rel=1e-12)
            else:
                assert second[section][key] == value


def test_probe_rejects_unverified_weights_before_model_allocation(local_inputs, tmp_path):
    inputs, constructed = local_inputs
    inputs["weights"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="official WavJEPA SHA-256"):
        probe.run_probe(**inputs, output=tmp_path / "report.json")
    assert not constructed
    assert not (tmp_path / "report.json").exists()


def test_probe_rejects_short_audio_without_padding_or_skipping(local_inputs, tmp_path):
    inputs, constructed = local_inputs
    write_wav(tmp_path / "recording-0.wav", [1, 2, 3, 4])
    with pytest.raises(ValueError, match="real samples"):
        probe.run_probe(**inputs, output=tmp_path / "report.json")
    assert not constructed
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"windows": 1},
        {"observed_windows": 1},
        {"observed_windows": 10},
        {"max_recordings": 1},
        {"seed": -1},
        {"window_batch_size": 0},
    ],
)
def test_probe_rejects_invalid_geometry_and_population(local_inputs, tmp_path, override):
    inputs, constructed = local_inputs
    with pytest.raises(ValueError):
        probe.run_probe(**inputs, output=tmp_path / "report.json", **override)
    assert not constructed


def test_existing_report_never_overwritten_or_model_allocated(local_inputs, tmp_path):
    inputs, constructed = local_inputs
    output = tmp_path / "report.json"
    output.write_text("keep me")
    with pytest.raises(FileExistsError):
        probe.run_probe(**inputs, output=output)
    assert output.read_text() == "keep me"
    assert not constructed


def test_atomic_publication_rejects_racing_existing_file_and_removes_temp(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("keep me")
    with pytest.raises(FileExistsError):
        probe.publish_report(output, {"status": "complete"})
    assert output.read_text() == "keep me"
    assert list(tmp_path.iterdir()) == [output]
    with pytest.raises(ValueError):
        probe.publish_report(tmp_path / "invalid.json", {"number": float("nan")})
    assert list(tmp_path.iterdir()) == [output]

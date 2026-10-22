"""Numerical contracts for the pinned-WavJEPA hierarchical dynamics extension.

These tiny randomly initialized models test implementation, not representation
quality, semantic abstraction, forecasting skill, or the published benchmark.
"""

import json
import struct
import wave
from copy import deepcopy
from functools import partial
from pathlib import Path

import lightning as L
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.dynamics_inference import load_model
from src.eval import evaluate
from src.models.components.hierarchy import HierarchyEncoder
from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.models.hierarchical_wavjepa_module import HierarchicalWavJEPAModule
from src.models.hwavjepa_module import HWavJEPAModule
from src.train import train
from src.utils.runtime import file_sha256

WINDOW_SAMPLES = 512
CONTEXT_WINDOWS = 5
HORIZONS = [1, 2, 4]


def model_factory(**overrides) -> HierarchicalWavJEPAModule:
    backbone = WavJEPABackbone(
        config={
            "sample_rate": 16_000,
            "window_samples": WINDOW_SAMPLES,
            "conv_layers": [[8, 4, 2], [8, 3, 2]],
            "encoder_dim": 16,
            "encoder_heads": 2,
            "encoder_layers": 1,
            "decoder_dim": 8,
            "decoder_heads": 2,
            "decoder_layers": 1,
            "average_top_k_layers": 1,
            "masker": {
                "target_masks_per_context": 1,
                "context_mask_length": 2,
                "target_length": 2,
                "context_mask_prob": 0.25,
                "target_prob": 0.25,
            },
        }
    )
    options = {
        "backbone": backbone,
        "optimizer": partial(torch.optim.AdamW, lr=0.001),
        "context_windows": CONTEXT_WINDOWS,
        "horizons": HORIZONS,
        "state_strides": [1, 2, 4],
        "hierarchy": True,
        "hidden_dim": 16,
        "predictor_layers": 1,
        "predictor_heads": 2,
        "prediction_weights": [1.0, 1.0, 1.0],
        "masked_weight": 1.0,
        "regularization_weight": 0.1,
        "ema_decay": 0.9,
        "eval_seed": 0,
        "sigreg_projections": 8,
    }
    return HierarchicalWavJEPAModule(**(options | overrides))


def waveform(batch_size=3):
    return torch.randn(batch_size, (CONTEXT_WINDOWS + max(HORIZONS)) * WINDOW_SAMPLES)


def gradient_magnitude(module):
    return sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def checkpoint_for(model):
    checkpoint = {"state_dict": deepcopy(model.state_dict())}
    model.on_save_checkpoint(checkpoint)
    return checkpoint


def test_hierarchical_loss_shapes_and_manual_horizon_alignment():
    model = model_factory().eval()
    audio = waveform()
    result = model.compute_loss(audio, seed=5)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["masked_loss"])
    assert len(result["prediction_terms"]) == 3
    assert len(result["regularization_terms"]) == 3
    assert len(result["levels"]) == 3
    targets = model.encode_targets(audio)
    for level, horizon in enumerate(HORIZONS):
        assert result["predictions"][level].shape == (3, 16)
        assert result["targets"][level].shape == (3, 16)
        assert not result["targets"][level].requires_grad
        expected = targets[level][:, CONTEXT_WINDOWS - 1 + horizon]
        torch.testing.assert_close(result["targets"][level], expected)
        manual_loss = torch.nn.functional.mse_loss(result["predictions"][level], expected)
        torch.testing.assert_close(result["prediction_terms"][level], manual_loss)


def test_high_level_prediction_gradient_reaches_both_abstractions_and_backbone():
    model = model_factory()
    result = model.compute_loss(waveform(), seed=13)
    result["prediction_terms"][-1].backward()
    assert gradient_magnitude(model.abstractions[0]) > 0
    assert gradient_magnitude(model.abstractions[1]) > 0
    assert gradient_magnitude(model.backbone.model.encoder) > 0
    assert gradient_magnitude(model.backbone.model.extract_audio) > 0
    assert all(p.grad is None for p in model.backbone.model.teacher_encoder.parameters())
    assert all(p.grad is None for p in model.teacher_abstractions.parameters())


def test_future_audio_cannot_change_any_prediction_even_through_normalization():
    model = model_factory().eval()
    audio = waveform()
    changed = audio.clone()
    cutoff = CONTEXT_WINDOWS * WINDOW_SAMPLES
    changed[:, cutoff:] = 19 * torch.randn_like(changed[:, cutoff:]) + 70
    first = model.compute_loss(audio, seed=17)
    second = model.compute_loss(changed, seed=17)
    past_only = model.predict(audio[:, :cutoff])
    for prediction, changed_prediction, direct in zip(
        first["predictions"], second["predictions"], past_only, strict=True
    ):
        torch.testing.assert_close(prediction, changed_prediction, rtol=0, atol=0)
        torch.testing.assert_close(prediction, direct, rtol=0, atol=0)
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(first["targets"], second["targets"], strict=True)
    )


def test_future_samples_have_zero_gradient_from_forecasting_predictions():
    model = model_factory().eval()
    audio = waveform().requires_grad_()
    result = model.compute_loss(audio, seed=19)
    result["predictions"][-1].square().sum().backward()
    cutoff = CONTEXT_WINDOWS * WINDOW_SAMPLES
    assert audio.grad is not None
    assert audio.grad[:, :cutoff].abs().sum() > 0
    assert torch.count_nonzero(audio.grad[:, cutoff:]) == 0


def test_all_strides_use_the_same_physical_context_origin_and_absolute_positions():
    model = model_factory().eval()
    audio = waveform()[:, : CONTEXT_WINDOWS * WINDOW_SAMPLES]
    levels = model.encode(audio)
    observed = []
    handles = [
        predictor.register_forward_pre_hook(lambda _module, args: observed.append(args))
        for predictor in model.predictors
    ]
    try:
        model.predict(audio)
    finally:
        for handle in handles:
            handle.remove()
    for level, stride, (states, positions) in zip(levels, [1, 2, 4], observed, strict=True):
        expected_positions = torch.arange(0, CONTEXT_WINDOWS, stride)
        assert torch.equal(positions, expected_positions)
        assert positions[0] == 0
        assert positions[-1] == CONTEXT_WINDOWS - 1
        torch.testing.assert_close(states, level[:, expected_positions])


def test_encoding_is_differentiable_and_preserves_completed_window_count():
    model = model_factory()
    audio = waveform()
    levels = model.encode(audio)
    assert len(levels) == 3
    assert all(level.shape == (3, 9, 16) for level in levels)
    assert all(level.requires_grad for level in levels)
    levels[-1].square().mean().backward()
    assert gradient_magnitude(model.backbone.model.encoder) > 0


@pytest.mark.parametrize("windows", [5, 8, 13])
@pytest.mark.parametrize("method", ["encode", "encode_targets"])
def test_subsampled_hierarchy_matches_dense_states_and_derives_counts(windows, method):
    model = model_factory().eval()
    audio = torch.randn(2, windows * WINDOW_SAMPLES)
    encode = getattr(model, method)
    dense = encode(audio)
    sampled = encode(audio, subsample=True)
    for full, sparse, stride in zip(dense, sampled, model.state_strides, strict=True):
        assert sparse.shape == (2, (windows - 1) // stride + 1, 16)
        torch.testing.assert_close(sparse, full[:, ::stride])


def test_skipped_windows_do_not_affect_upper_states_or_receive_their_gradients():
    model = model_factory().eval()
    audio = torch.randn(2, 9, WINDOW_SAMPLES)
    changed = audio.clone()
    # The high-level state grid uses windows 0, 4, 8. Pooling would mix in others.
    skipped = [1, 2, 3, 5, 6, 7]
    changed[:, skipped] = torch.randn_like(changed[:, skipped]) * 20 + 70
    original_high = model.encode(audio.flatten(1), subsample=True)[-1]
    changed_high = model.encode(changed.flatten(1), subsample=True)[-1]
    torch.testing.assert_close(original_high, changed_high, rtol=0, atol=0)
    audio.requires_grad_()
    model.encode(audio.flatten(1), subsample=True)[-1].square().sum().backward()
    assert audio.grad[:, [0, 4, 8]].abs().sum() > 0
    assert torch.count_nonzero(audio.grad[:, skipped]) == 0


def test_flat_control_uses_one_state_space_at_all_three_horizons():
    model = model_factory(hierarchy=False, state_strides=[1, 1, 1]).eval()
    audio = waveform()
    levels = model.encode(audio)
    for level in levels[1:]:
        torch.testing.assert_close(level, levels[0], rtol=0, atol=0)
    result = model.compute_loss(audio, seed=23)
    assert len(result["predictions"]) == 3
    targets = model.encode_targets(audio)[0]
    for prediction, target, horizon in zip(
        result["predictions"], result["targets"], HORIZONS, strict=True
    ):
        assert prediction.shape == (3, 16)
        torch.testing.assert_close(target, targets[:, CONTEXT_WINDOWS - 1 + horizon])


def test_collapsed_upper_representations_incur_regularization_penalty():
    model = model_factory().eval()
    with torch.no_grad():
        for parameter in model.abstractions.parameters():
            parameter.zero_()
    result = model.compute_loss(waveform(), seed=29)
    assert result["regularization_terms"][1] > 0
    assert result["regularization_terms"][2] > 0


@pytest.mark.parametrize("samples", [0, 1, 512, 9 * 512 - 1, 9 * 512 + 1])
def test_training_rejects_wrong_sample_count(samples):
    with pytest.raises(ValueError):
        model_factory().compute_loss(torch.randn(2, samples), seed=0)


@pytest.mark.parametrize("windows", [1, 4, 6, 9])
def test_predict_requires_exactly_past_context_not_future_windows(windows):
    with pytest.raises(ValueError):
        model_factory().predict(torch.randn(2, windows * WINDOW_SAMPLES))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_audio_is_rejected(invalid):
    model = model_factory()
    audio = waveform()
    audio[0, 0] = invalid
    with pytest.raises(ValueError):
        model.compute_loss(audio, seed=0)


@pytest.mark.parametrize(
    "options",
    [
        {"horizons": [1, 2]},
        {"horizons": [0, 2, 4]},
        {"horizons": [1, 1, 4]},
        {"horizons": [1, 4, 2]},
        {"horizons": [1, 2, 4.5]},
        {"state_strides": [1, 2]},
        {"state_strides": [1, 0, 4]},
        {"state_strides": [1, 2, 4.5]},
        {"context_windows": 7, "state_strides": [1, 2, 3]},
        {"context_windows": 0},
        {"context_windows": True},
        {"ema_decay": 1.0},
    ],
)
def test_invalid_horizon_stride_and_training_contracts_are_rejected(options):
    with pytest.raises(ValueError):
        model_factory(**options)


def test_explicit_seed_repeats_masking_predictions_and_regularization():
    model = model_factory().eval()
    audio = waveform()
    first = model.compute_loss(audio, seed=31)
    torch.rand(100)
    second = model.compute_loss(audio, seed=31)
    for key in ("loss", "masked_loss"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    for left, right in zip(
        first["regularization_terms"], second["regularization_terms"], strict=True
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_seeded_evaluation_does_not_advance_the_global_torch_rng():
    model = model_factory().eval()
    audio = waveform()
    before = torch.get_rng_state().clone()
    model.compute_loss(audio)
    assert torch.equal(before, torch.get_rng_state())


def test_teacher_parameters_stay_frozen_and_outside_optimizer_in_train_mode():
    model = model_factory().train()
    assert not model.backbone.model.teacher_encoder.training
    assert not model.teacher_abstractions.training
    teachers = [
        *model.backbone.model.teacher_encoder.parameters(),
        *model.teacher_abstractions.parameters(),
    ]
    assert all(not parameter.requires_grad for parameter in teachers)
    optimizer = model.configure_optimizers()
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not {id(p) for p in teachers} & optimized_ids


def test_teacher_update_matches_ema_for_encoder_and_upper_abstractions():
    model = model_factory()
    pairs = [
        (model.backbone.model.encoder, model.backbone.model.teacher_encoder),
        *zip(model.abstractions, model.teacher_abstractions, strict=True),
    ]
    expected = []
    with torch.no_grad():
        for online, teacher in pairs:
            for student_parameter, teacher_parameter in zip(
                online.parameters(), teacher.parameters(), strict=True
            ):
                student_parameter.add_(0.5)
                expected.append(
                    (teacher_parameter, 0.9 * teacher_parameter + 0.1 * student_parameter)
                )
    model.update_teacher()
    assert model.ema_updates.item() == 1
    for actual, target in expected:
        torch.testing.assert_close(actual, target)


def test_optimizer_updates_teacher_only_after_successful_step():
    model = model_factory()
    optimizer = model.configure_optimizers()
    model.compute_loss(waveform(), seed=37)["loss"].backward()
    assert model.ema_updates.item() == 0
    model.on_before_optimizer_step(optimizer)
    optimizer.step()
    assert model.ema_updates.item() == 1
    assert optimizer.state


def test_checkpoint_round_trip_preserves_predictions_and_plain_metadata(tmp_path):
    source = model_factory().eval()
    checkpoint = checkpoint_for(source)
    metadata = source.checkpoint_metadata()
    assert json.loads(json.dumps(metadata)) == metadata
    path = tmp_path / "hierarchical.ckpt"
    torch.save(checkpoint, path)
    saved = torch.load(path, weights_only=True, map_location="cpu")
    restored = model_factory().eval()
    restored.on_load_checkpoint(saved)
    restored.load_state_dict(saved["state_dict"], strict=True)
    audio = waveform()[:, : CONTEXT_WINDOWS * WINDOW_SAMPLES]
    for actual, expected in zip(restored.predict(audio), source.predict(audio), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_explicit_inference_loader_reconstructs_encoder_predictions_and_teacher(tmp_path):
    source = model_factory().eval()
    source.update_teacher()
    path = tmp_path / "dynamics.ckpt"
    torch.save(checkpoint_for(source), path)
    restored = load_model(path)
    assert isinstance(restored, HierarchicalWavJEPAModule)
    assert not restored.training
    assert next(restored.parameters()).device.type == "cpu"
    assert restored.checkpoint_metadata() == source.checkpoint_metadata()
    assert restored.ema_updates.item() == 1
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
    audio = waveform()
    with torch.no_grad():
        for method, samples in (
            ("encode", audio),
            ("encode_targets", audio),
            ("predict", audio[:, : CONTEXT_WINDOWS * WINDOW_SAMPLES]),
        ):
            actual = getattr(restored, method)(samples)
            expected = getattr(source, method)(samples)
            for actual_level, expected_level in zip(actual, expected, strict=True):
                # Frozen inference may select a different native FP32 kernel.
                torch.testing.assert_close(actual_level, expected_level, rtol=1e-5, atol=1e-6)


def test_inference_loader_rejects_missing_metadata(tmp_path):
    path = tmp_path / "no-metadata.ckpt"
    torch.save({"state_dict": model_factory().state_dict()}, path)
    with pytest.raises(ValueError, match="missing or invalid.*metadata"):
        load_model(path)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_inference_loader_rejects_nonfinite_weights(tmp_path, invalid):
    checkpoint = checkpoint_for(model_factory())
    parameter = checkpoint["state_dict"]["abstractions.0.0.weight"]
    parameter[0, 0] = invalid
    path = tmp_path / "nonfinite.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="nonfinite state"):
        load_model(path)


def test_inference_loader_rejects_legacy_model_family(tmp_path):
    legacy = HWavJEPAModule(
        encoder=HierarchyEncoder(patch_size=8, widths=(8, 12, 16), pool_factors=(2, 2)),
        optimizer=partial(torch.optim.AdamW, lr=0.001),
        protected_dim=8,
        input_sample_rate=16_000,
    )
    checkpoint = {"state_dict": legacy.state_dict()}
    legacy.on_save_checkpoint(checkpoint)
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="missing or invalid.*metadata"):
        load_model(path)


@pytest.mark.parametrize("invalid", [None, [], "not-model-state"])
def test_inference_loader_rejects_nondictionary_checkpoint(tmp_path, invalid):
    path = tmp_path / "invalid.ckpt"
    torch.save(invalid, path)
    with pytest.raises(ValueError, match="checkpoint dictionary"):
        load_model(path)


def test_inference_loader_rejects_nontensor_model_state(tmp_path):
    checkpoint = checkpoint_for(model_factory())
    checkpoint["state_dict"]["unexpected-object"] = {"configuration": "not weights"}
    path = tmp_path / "invalid-state.ckpt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="tensor-only state_dict"):
        load_model(path)


@pytest.mark.parametrize(
    "changed", [{"horizons": [1, 3, 4]}, {"ema_decay": 0.8}, {"context_windows": 9}]
)
def test_checkpoint_rejects_changed_semantics_even_when_parameter_shapes_match(changed):
    source = model_factory()
    altered = model_factory(**changed)
    with pytest.raises(ValueError):
        altered.on_load_checkpoint(checkpoint_for(source))


@pytest.mark.parametrize("corruption", ["missing", "version", "source", "nan"])
def test_checkpoint_rejects_missing_unknown_or_corrupted_provenance(corruption):
    model = model_factory()
    checkpoint = checkpoint_for(model)
    key = model.CHECKPOINT_KEY
    if corruption == "missing":
        checkpoint.pop(key)
    elif corruption == "version":
        checkpoint[key]["schema_version"] = 999
    elif corruption == "source":
        checkpoint[key]["upstream_commit"] = "another-source-revision"
    else:
        checkpoint[key]["dynamics"]["masked_weight"] = float("nan")
    with pytest.raises(ValueError):
        model.on_load_checkpoint(checkpoint)


@pytest.mark.integration
def test_lightning_nonfinite_gradient_aborts_optimizer_and_ema(tmp_path):
    model = model_factory()
    parameter = next(model.abstractions[1].parameters())
    handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    before = deepcopy(model.state_dict())
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    loader = DataLoader([{"waveform": item} for item in waveform(2)], batch_size=2)
    try:
        with pytest.raises(FloatingPointError, match="non-finite gradient"):
            trainer.fit(model, train_dataloaders=loader)
    finally:
        handle.remove()
    assert trainer.optimizers[0].state == {}
    assert model.ema_updates.item() == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.integration
def test_lightning_accumulation_and_checkpoint_resume_update_ema_once_per_step(tmp_path):
    model = model_factory()
    loader = DataLoader([{"waveform": item} for item in waveform(batch_size=8)], batch_size=2)
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "accumulate_grad_batches": 2,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "default_root_dir": tmp_path,
    }
    trainer = L.Trainer(max_steps=2, **options)
    trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == 2
    assert model.ema_updates.item() == 2
    checkpoint = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = model_factory()
    resumed = L.Trainer(max_steps=3, **options)
    resumed.fit(restored, train_dataloaders=loader, ckpt_path=str(checkpoint))
    assert resumed.global_step == 3
    assert restored.ema_updates.item() == 3


@pytest.mark.integration
def test_pcm_wav_hydra_train_evaluate_and_inference_pipeline(make_config, tmp_path):
    """Generated noise WAVs exercise assembly and provenance, not audio quality."""
    num_samples = (CONTEXT_WINDOWS + max(HORIZONS)) * WINDOW_SAMPLES
    manifests = {}
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for index in range(3):
            path = tmp_path / f"{split}-{index}.wav"
            generator = torch.Generator().manual_seed(100 + split_index * 3 + index)
            samples = torch.randint(-12_000, 12_001, (num_samples,), generator=generator)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(struct.pack(f"<{num_samples}h", *samples.tolist()))
            rows.append({"path": path.name, "recording_id": f"{split}-recording-{index}"})
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        manifests[f"{split}_manifest"] = str(manifest)

    def pipeline_config(output, name="train", checkpoint=None):
        overrides = [
            "experiment=upstream_smoke",
            "seed=41",
            "logger=[]",
            f"paths.output_dir={output}",
        ]
        if name == "train":
            overrides.append("test=true")
        if checkpoint is not None:
            overrides.append(f"ckpt_path={checkpoint}")
        cfg = make_config(overrides, name=name)
        cfg.data = OmegaConf.create(
            {
                "_target_": "src.data.wav_datamodule.WAVManifestDataModule",
                **manifests,
                "sample_rate": 16_000,
                "num_samples": num_samples,
                "batch_size": 2,
                "minimum_batch_size": 2,
                "require_full_length": True,
                "num_workers": 0,
            }
        )
        return cfg

    output = tmp_path / "wav-training"
    metrics, objects = train(pipeline_config(output))
    model = objects["model"]
    assert isinstance(model, HierarchicalWavJEPAModule)
    assert objects["trainer"].global_step == 1
    assert model.ema_updates.item() == 1
    for stage in ("train", "val", "test"):
        loader = getattr(objects["datamodule"], f"{stage}_dataloader")()
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["waveform"].shape == (3, num_samples)
        assert torch.isfinite(metrics[f"{stage}/loss"])

    checkpoint_path = Path(objects["trainer"].checkpoint_callback.best_model_path)
    assert (output / "checkpoints" / "last.ckpt").is_file()
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    assert checkpoint[model.CHECKPOINT_KEY] == model.checkpoint_metadata()
    assert "hwavjepa" not in checkpoint
    record = json.loads((output / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["global_step"] == 1
    assert record["checkpoints"]["best"]["sha256"] == file_sha256(checkpoint_path)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["model_contract"] == model.checkpoint_metadata()
    assert provenance["upstream_initialization"]["weights"] == "random"
    for key, path in manifests.items():
        assert provenance["manifests"][key]["sha256"] == file_sha256(path)

    evaluation_output = tmp_path / "wav-evaluation"
    eval_metrics, evaluated = evaluate(
        pipeline_config(evaluation_output, name="eval", checkpoint=checkpoint_path)
    )
    assert torch.isfinite(eval_metrics["test/loss"])
    assert json.loads((evaluation_output / "run.json").read_text())["status"] == "completed"
    eval_provenance = json.loads((evaluation_output / "provenance.json").read_text())
    assert eval_provenance["model_contract"] == model.checkpoint_metadata()
    assert eval_provenance["input_checkpoint_sha256"] == file_sha256(checkpoint_path)
    restored = load_model(checkpoint_path)
    for candidate in (evaluated["model"], restored):
        for key, expected in checkpoint["state_dict"].items():
            torch.testing.assert_close(candidate.state_dict()[key], expected, rtol=0, atol=0)
    evaluation_batches = list(evaluated["datamodule"].test_dataloader())
    assert len(evaluation_batches) == 1
    audio = evaluation_batches[0]["waveform"]
    assert audio.shape == (3, num_samples)
    prefix = audio[:, : CONTEXT_WINDOWS * WINDOW_SAMPLES]
    evaluated["model"].eval()
    with torch.no_grad():
        for actual, expected in zip(
            restored.predict(prefix), evaluated["model"].predict(prefix), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

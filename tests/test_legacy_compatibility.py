"""Historical prototype imports/configs remain explicit compatibility interfaces."""

import sys

import pytest
from omegaconf import OmegaConf

from src import embed
from src.legacy.components.hierarchy import HierarchyEncoder, patchify
from src.legacy.components.masking import grouped_patch_mask
from src.legacy.components.predictor import MaskedPredictor
from src.legacy.hwavjepa_module import HWavJEPAModule
from src.legacy.losses import feature_std, masked_prediction_loss, reference_reconstruction_loss
from src.models.components import hierarchy, masking, predictor
from src.models.hwavjepa_module import HWavJEPAModule as CompatibleModule
from src.objectives import losses


def test_previous_import_paths_reexport_identical_objects():
    assert CompatibleModule is HWavJEPAModule
    assert hierarchy.HierarchyEncoder is HierarchyEncoder
    assert hierarchy.patchify is patchify
    assert masking.grouped_patch_mask is grouped_patch_mask
    assert predictor.MaskedPredictor is MaskedPredictor
    assert losses.feature_std is feature_std
    assert losses.masked_prediction_loss is masked_prediction_loss
    assert losses.reference_reconstruction_loss is reference_reconstruction_loss
    assert HWavJEPAModule.__module__ == "src.legacy.hwavjepa_module"


@pytest.mark.parametrize(
    "name",
    ["smoke", "audio", "hwavjepa", "hwavjepa_unprotected", "flat", "flat_protected", "two_level"],
)
def test_archived_experiment_and_previous_alias_resolve_identically(make_config, name):
    legacy = make_config([f"experiment=legacy/{name}"])
    compatible = make_config([f"experiment={name}"])
    assert OmegaConf.to_container(legacy, resolve=True) == OmegaConf.to_container(
        compatible, resolve=True
    )


@pytest.mark.parametrize("family", [None, "legacy"])
def test_embedding_cli_defaults_to_dynamics_and_retains_explicit_legacy(monkeypatch, family):
    arguments = [
        "src.embed",
        "--checkpoint",
        "model.ckpt",
        "--manifest",
        "test.jsonl",
        "--output",
        "features",
        "--num-samples",
        "16160",
    ]
    if family is not None:
        arguments.extend(["--model-family", family])
    monkeypatch.setattr(sys, "argv", arguments)
    captured = []

    def extract(**options):
        captured.append(options)
        return {"status": "test"}

    monkeypatch.setattr(embed, "extract_embeddings", extract)
    embed.main()
    assert captured[0]["model_family"] == (family or "dynamics")

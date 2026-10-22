"""Compare against a separate, hash-verified checkout of the original WavJEPA core.

Usage: python -m src.verify_upstream --upstream /path/to/wavjepa [--pretrained]

No reference code is downloaded or rewritten. Package import shims bypass the
unrelated denoiser initializer and the upstream types initializer's alias typo;
the actual model, frontend, normalization and loss source files are unchanged.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import lightning.pytorch
import numpy as np
import torch

from src.models.components.wavjepa_backbone import WavJEPABackbone
from src.pretrained import file_sha256


def load_reference(source: Path):
    manifest = json.loads((Path(__file__).parent / "vendor/wavjepa/UPSTREAM.json").read_text())
    for relative, digest in manifest["source_sha256"].items():
        if file_sha256(source / relative) != digest:
            raise ValueError(f"reference source does not match pinned upstream: {relative}")
    if any(name == "wavjepa" or name.startswith("wavjepa.") for name in sys.modules):
        raise RuntimeError(
            "run verification in a fresh process without an imported wavjepa package"
        )
    # Only import routing is adapted; all source method bodies remain original.
    for name, path in (
        ("wavjepa", source / "wavjepa"),
        ("wavjepa.extractors", source / "wavjepa/extractors"),
        ("wavjepa.types", source / "wavjepa/types"),
    ):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    configs = importlib.import_module("wavjepa.types.wavjepa_configs")
    for name in ("ForwardReturn", "TransformerLayerCFG", "TransformerEncoderCFG"):
        setattr(sys.modules["wavjepa.types"], name, getattr(configs, name))
    sys.modules["pytorch_lightning"] = lightning.pytorch
    model_type = importlib.import_module("wavjepa.jepa").JEPA
    extractor_type = importlib.import_module(
        "wavjepa.extractors.audio_feature_extractor"
    ).ConvFeatureExtractor
    return model_type, extractor_type, configs


def verify(source: Path, *, pretrained: bool = False) -> dict:
    """Check tokens, teacher targets, masked loss and (small model) gradients."""
    model_type, extractor_type, types = load_reference(source)
    torch.set_num_threads(1)
    torch.manual_seed(24)
    config = (
        None
        if pretrained
        else {
            "window_samples": 512,
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
                "context_mask_prob": 0.25,
                "context_mask_length": 2,
                "target_prob": 0.25,
                "target_length": 2,
            },
        }
    )
    backbone = WavJEPABackbone(config, pretrained=pretrained, local_files_only=True).eval()
    cfg = backbone.config
    reference = model_type(
        feature_extractor=extractor_type(
            conv_layers_spec=[tuple(spec) for spec in cfg["conv_layers"]],
            in_channels=1,
            depthwise=False,
        ),
        transformer_encoder_layers_cfg=types.TransformerLayerCFG.create(
            d_model=cfg["encoder_dim"], nhead=cfg["encoder_heads"], mlp_ratio=cfg["mlp_ratio"]
        ),
        transformer_encoder_cfg=types.TransformerEncoderCFG.create(
            num_layers=cfg["encoder_layers"]
        ),
        transformer_decoder_layers_cfg=types.TransformerLayerCFG.create(
            d_model=cfg["decoder_dim"], nhead=cfg["decoder_heads"], mlp_ratio=cfg["mlp_ratio"]
        ),
        transformer_decoder_cfg=types.TransformerEncoderCFG.create(
            num_layers=cfg["decoder_layers"]
        ),
        average_top_k_layers=cfg["average_top_k_layers"],
        resample_sr=cfg["sample_rate"],
        process_audio_seconds=cfg["window_samples"] / cfg["sample_rate"],
        nr_samples_per_audio=1,
        compile_modules=False,
        use_gradient_checkpointing=False,
    ).eval()
    reference.load_state_dict(backbone.model.state_dict(), strict=True)
    wave = torch.randn(2, 1, cfg["window_samples"])
    # Match the original crop normalization explicitly; do not call the adapter
    # to produce the reference input or use its methods for reference outputs.
    normalized = (wave - wave.mean(dim=(-2, -1), keepdim=True)) / (
        wave.std(dim=(-2, -1), keepdim=True) + 1e-5
    )
    with torch.no_grad():
        expected_tokens = reference.get_audio_representation(normalized, None)
        tokens = backbone.encode_tokens(wave)[:, 0]
        torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    masks = backbone.masker(
        2, backbone.model.total_patches, 1, generator=np.random.default_rng(123)
    )
    # Predetermined masks isolate model equivalence from the intentional local
    # RNG API adaptation in masking. The two implementations receive same masks.
    with torch.set_grad_enabled(not pretrained):
        actual = backbone.model(normalized, *masks)
        expected = reference(normalized, *masks)
        for key in ("loss", "preds", "targets"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if not pretrained:
            actual["loss"].backward()
            expected["loss"].backward()
            for (name, left), (other, right) in zip(
                backbone.model.named_parameters(), reference.named_parameters(), strict=True
            ):
                assert name == other
                if left.grad is None or right.grad is None:
                    assert left.grad is None and right.grad is None, name
                else:
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)
    return {
        "source_commit": backbone.source_commit,
        "weights_sha256": backbone.checkpoint_sha256,
        "pretrained": pretrained,
        "tokens_shape": list(tokens.shape),
        "max_token_difference": (tokens - expected_tokens).abs().max().item(),
        "masked_loss": actual["loss"].item(),
        "matched": ["tokens", "predictions", "teacher_targets", "masked_loss"]
        + ([] if pretrained else ["parameter_gradients"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.upstream.resolve(), pretrained=args.pretrained), indent=2))


if __name__ == "__main__":
    main()

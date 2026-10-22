"""Read-only validation of real WAV manifests before scheduling training.

Run ``python -m src.validate_data --train train.jsonl --val val.jsonl --test test.jsonl``.
"""

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.data.wav_datamodule import validate_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training JSONL manifest")
    parser.add_argument("--val", required=True, help="Validation JSONL manifest")
    parser.add_argument("--test", required=True, help="Test JSONL manifest")
    parser.add_argument(
        "--experiment", help="Use this Hydra experiment's WAV length/batch settings"
    )
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument(
        "--require-full-length", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--batch-size", type=int, help="Batch size per process/device")
    parser.add_argument(
        "--minimum-batch-size", type=int, help="Minimum batch size per process/device"
    )
    parser.add_argument("--world-size", type=int, help="Total training processes/devices")
    parser.add_argument(
        "--drop-last-train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require at least one full training batch on every device",
    )
    args = parser.parse_args()
    try:
        settings = {
            "sample_rate": 16_000,
            "num_samples": 1024,
            "require_full_length": False,
            "batch_size": 4,
            "minimum_batch_size": 1,
            "world_size": 1,
            "train_drop_last": False,
        }
        if args.experiment:
            root = Path(__file__).resolve().parents[1]
            with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
                cfg = compose(config_name="train", overrides=[f"experiment={args.experiment}"])
            if cfg.data._target_ != "src.data.wav_datamodule.WAVManifestDataModule":
                raise ValueError("--experiment must select a WAVManifestDataModule experiment")
            for name in settings.keys() - {"world_size"}:
                settings[name] = OmegaConf.select(cfg, f"data.{name}", default=settings[name])
            devices = cfg.trainer.devices
            if args.world_size is None:
                if isinstance(devices, int) and devices >= 1:
                    settings["world_size"] = devices * OmegaConf.select(
                        cfg, "trainer.num_nodes", default=1
                    )
                elif OmegaConf.is_list(devices) and len(devices) >= 1:
                    settings["world_size"] = len(devices) * OmegaConf.select(
                        cfg, "trainer.num_nodes", default=1
                    )
                else:
                    raise ValueError(
                        "Use --world-size when experiment trainer.devices is not an explicit count/list"
                    )
        for name in settings:
            value = getattr(args, "drop_last_train" if name == "train_drop_last" else name)
            if value is not None:
                settings[name] = value
        report = validate_manifests(args.train, args.val, args.test, **settings)
    except (ValueError, HydraException, OmegaConfBaseException) as error:
        print(json.dumps({"valid": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

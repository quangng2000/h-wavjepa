"""Generate tiny synthetic HEAR exports; these are integration fixtures, not scores.

Run from the repository root in the training environment:
  python -m evaluation.hear.make_smoke --output artifacts/hear-smoke
Then run evaluation/hear/run.py in the separate evaluation environment.
"""

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize_config_dir

from src.hear_api import load_model
from src.hear_eval import export_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config_root = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
        config = compose(config_name="train", overrides=["experiment=upstream_smoke"])
    encoder = hydra.utils.instantiate(config.model).eval()
    checkpoint = {"state_dict": encoder.state_dict()}
    encoder.on_save_checkpoint(checkpoint)
    checkpoint_path = root / "tiny-random.ckpt"
    torch.save(checkpoint, checkpoint_path)

    for kind, representation in (("scene", "high"), ("event", "backbone")):
        task = root / f"synthetic-{kind}"
        task.mkdir()
        metadata = {
            "task_name": f"synthetic_{kind}_integration_only",
            "embedding_type": kind,
            "prediction_type": "multiclass" if kind == "scene" else "multilabel",
            "sample_duration": None,
            "split_mode": "trainvaltest",
            "splits": ["train", "valid", "test"],
            "evaluation": ["top1_acc"] if kind == "scene" else ["event_onset_50ms_fms"],
            "evaluation_params": {
                # This short grid is only for generated integration fixtures.
                "task_specific_param_grid": {
                    "hidden_layers": [0],
                    "hidden_dim": [16],
                    "dropout": [0.0],
                    "batch_size": [4 if kind == "scene" else 512],
                    "max_epochs": [2],
                    "check_val_every_n_epoch": [1],
                    "patience": [1],
                    "lr": [0.001],
                },
                "event_postprocessing_grid": {"median_filter_ms": [0], "min_duration": [0]},
            },
        }
        (task / "task_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (task / "labelvocabulary.csv").write_text("idx,label\n0,low\n1,high\n")
        for split_index, split in enumerate(metadata["splits"]):
            folder = task / "16000" / split
            folder.mkdir(parents=True)
            labels = {}
            for index in range(8):
                # Independent lengths/phases in each split; no waveform reuse.
                count = 4096 + split_index * 73 + index * 19
                samples = np.arange(count, dtype=np.float32) / 16000
                frequency = 220 if index % 2 == 0 else 880
                waveform = 0.5 * np.sin(2 * np.pi * frequency * samples + split_index + index)
                filename = f"{split}-{index}.wav"
                sf.write(folder / filename, waveform, 16000, subtype="PCM_16")
                label = "low" if index % 2 == 0 else "high"
                labels[filename] = (
                    [label]
                    if kind == "scene"
                    else [{"start": 0.0, "end": count * 1000 / 16000, "label": label}]
                )
            (task / f"{split}.json").write_text(json.dumps(labels, indent=2) + "\n")
        output = export_task(
            load_model(checkpoint_path, representation=representation, window_batch_size=4),
            task,
            root / f"export-{kind}",
            checkpoint_path=checkpoint_path,
        )
        print(output)


if __name__ == "__main__":
    main()

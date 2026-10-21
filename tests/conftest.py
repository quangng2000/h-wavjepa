from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir


@pytest.fixture(autouse=True)
def cpu_thread_limit():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def make_config(tmp_path):
    root = Path(__file__).resolve().parents[1]

    def factory(overrides=(), name="train"):
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            return compose(
                config_name=name,
                overrides=[
                    "experiment=smoke",
                    f"paths.root_dir={root}",
                    f"paths.output_dir={tmp_path / 'run'}",
                    "trainer.enable_model_summary=false",
                    *overrides,
                ],
            )

    return factory

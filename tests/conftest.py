import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ser.config import Config, DatasetConfig  # noqa: E402
from ser.synthesis import build_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    from ser.config import AudioConfig, EventConfig

    return Config(
        audio=AudioConfig(),
        events=EventConfig(),
        dataset=DatasetConfig(n_train=4, n_val=2, n_test=2,
                              n_interference=2, seed=1234),
    )


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory, tiny_config):
    root = tmp_path_factory.mktemp("data")
    build_dataset(root, tiny_config.dataset)
    return root


@pytest.fixture(scope="session")
def trained_pipeline(tmp_path_factory, tiny_config, data_dir):
    from ser.pipeline import Pipeline

    exp = tmp_path_factory.mktemp("exp")
    pipe = Pipeline(config=tiny_config, data_root=data_dir,
                    experiments_root=exp, model_dir=exp / "model")
    pipe.train()
    return pipe

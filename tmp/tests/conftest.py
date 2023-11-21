import pytest
import shutil
from pathlib import Path


@pytest.fixture
def global_datadir(tmp_path):
    original_shared_path = Path(__file__).parent.joinpath("data")
    temp_path = tmp_path / "data"
    shutil.copytree(original_shared_path, str(temp_path))
    return temp_path


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="run-slow option is not enabled")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

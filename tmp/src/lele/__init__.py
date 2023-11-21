__version__ = "dirty"
from pathlib import Path

_CONFIG_DIR = Path(__file__).parents[1].joinpath("hydra_configs").as_posix()
_HYDRA_VERSION = "1.3"
del Path

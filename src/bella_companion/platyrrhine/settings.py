from pathlib import Path

import pandas as pd

from bella_companion.settings import settings

THIS_DIR = Path(__file__).parent
DATA_DIR = THIS_DIR / "data"
TREE_FILE = DATA_DIR / "trees.nwk"
CHANGE_TIMES_FILE = DATA_DIR / "change_times.csv"
TRAITS_FILE = DATA_DIR / "traits.csv"
BEAST_CONFIG_PATH = THIS_DIR / "beast_config.xml"

TYPES = list(range(4))

CHANGE_TIMES: list[float] = (
    pd.read_csv(CHANGE_TIMES_FILE, header=None).values.flatten().tolist()  # pyright: ignore
)
TIME_BINS = [0, *CHANGE_TIMES]
N_TIME_BINS = len(TIME_BINS)
TIME_AXIS = list(reversed([0.0, *CHANGE_TIMES, 30.0]))

BELLA_CONFIGS = settings.bella_model_configs["BELLA-3"]
RATE_UPPER = 5

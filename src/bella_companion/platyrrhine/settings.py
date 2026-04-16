from pathlib import Path

import pandas as pd

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

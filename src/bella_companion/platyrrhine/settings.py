from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TREE_FILE = DATA_DIR / "trees.nwk"
CHANGE_TIMES_FILE = DATA_DIR / "change_times.csv"
TRAITS_FILE = DATA_DIR / "traits.csv"

TYPES = list(range(4))
CHANGE_TIMES: list[float] = (
    pd.read_csv(CHANGE_TIMES_FILE, header=None).values.flatten().tolist()  # pyright: ignore
)

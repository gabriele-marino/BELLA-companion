import os
import subprocess
from glob import glob
from itertools import chain
from pathlib import Path

import joblib
from tqdm import tqdm

from bella_companion.backend import read_weights_dir, summarize_logs_dir
from bella_companion.platyrrhine.settings import CHANGE_TIMES, TYPES


def summarize_platyrrhine():
    logs_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "platyrrhine"

    summaries = summarize_logs_dir(
        logs_dir=logs_dir,
        target_columns=[
            f"{rate}RateSPi{i}_{t}"
            for rate in ["birth", "death"]
            for i in range(len(CHANGE_TIMES) + 1)
            for t in TYPES
        ],
    )
    weights = read_weights_dir(logs_dir)

    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"], "platyrrhine")
    os.makedirs(summaries_dir, exist_ok=True)
    summaries.to_csv(summaries_dir / "BELLA.csv")
    joblib.dump(weights, summaries_dir / "BELLA.weights.pkl")

    options = [
        ("-log", tree_file) for tree_file in tqdm(glob(str(logs_dir / "*.trees")))
    ]
    combined_trees_file = summaries_dir / ".trees.combined.tmp.nwk"
    subprocess.run(
        ["logcombiner", *list(chain(*options)), "-o", str(combined_trees_file)]
    )

    subprocess.run(
        [
            "treeannotator",
            "-file",
            str(combined_trees_file),
            str(summaries_dir / "mcc.nexus"),
            "-burnin",
            "0",
            "-height",
            "median",
        ]
    )

    os.remove(combined_trees_file)

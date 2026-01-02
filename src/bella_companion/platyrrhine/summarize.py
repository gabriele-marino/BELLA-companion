import os
import subprocess
from glob import glob
from pathlib import Path

import joblib
from phylogenie import dump_newick
from tqdm import tqdm

from bella_companion.backend import (
    load_nexus_with_burnin,
    read_weights_dir,
    summarize_logs_dir,
)
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

    trees = [
        tree
        for tree_file in tqdm(glob(str(logs_dir / "*.trees")), "Summarizing trees")
        for tree in load_nexus_with_burnin(tree_file)
    ]
    trees_file = summaries_dir / ".trees.tmp.nwk"
    dump_newick(trees, trees_file)

    subprocess.run(
        [
            "treeannotator",
            "-file",
            str(trees_file),
            str(summaries_dir / "mcc.nexus"),
            "-burnin",
            "0",
            "-height",
            "median",
        ]
    )

    os.remove(trees_file)

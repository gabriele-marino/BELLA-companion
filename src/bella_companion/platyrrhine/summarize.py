import os
import shutil
from glob import glob
from pathlib import Path

import joblib
import pandas as pd
from phylogenie import Tree, dump_newick, load_nexus
from tqdm import tqdm

from bella_companion.backend import read_weights_dir, summarize_logs_dir


def summarize():
    data_dir = Path(__file__).parent / "data"
    change_times = pd.read_csv(data_dir / "change_times.csv", header=None)  # pyright: ignore
    n_time_bins = len(change_times) + 1

    logs_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "platyrrhine"
    summaries = summarize_logs_dir(
        logs_dir=logs_dir,
        target_columns=[
            f"{rate}RateSPi{i}_{t}"
            for rate in ["birth", "death"]
            for i in range(n_time_bins)
            for t in ["0", "1", "2", "3"]
        ],
    )
    weights = read_weights_dir(logs_dir)

    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"], "platyrrhine")
    os.makedirs(summaries_dir, exist_ok=True)
    summaries.to_csv(summaries_dir / "BELLA.csv")
    joblib.dump(weights, summaries_dir / "BELLA.weights.pkl")

    # trees: list[Tree] = []
    # for tree_file in tqdm(glob(str(logs_dir / "*.trees")), "Summarizing trees"):
    #    trees.extend(list(load_nexus(tree_file).values())[-10:])
    # dump_newick(trees, summaries_dir / "trees.nwk")

    # shutil.copy(logs_dir / "0.trees", summaries_dir / "sample-trees.nexus")

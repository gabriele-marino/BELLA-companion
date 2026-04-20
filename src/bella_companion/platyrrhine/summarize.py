import json
import os
import subprocess
from glob import glob
from itertools import chain
from pathlib import Path

import joblib
from tqdm import tqdm

from bella_companion.backend import read_weights_dir, summarize_logs_dir
from bella_companion.platyrrhine.run import JOB_IDS_FILENAME
from bella_companion.platyrrhine.settings import CHANGE_TIMES, TYPES
from bella_companion.settings import BELLA_REFERENCE_MODEL, BELLA_SETTINGS


def summarize_platyrrhine():
    base_logs_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "platyrrhine"
    beast_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "platyrrhine"
    os.makedirs(summaries_dir, exist_ok=True)

    with open(beast_output_dir / JOB_IDS_FILENAME, "r") as f:
        job_ids: dict[str, dict[str, str]] = json.load(f)

    for model in BELLA_SETTINGS:
        logs_dir = base_logs_dir / model
        summaries = summarize_logs_dir(
            logs_dir=logs_dir,
            target_columns=[
                f"{rate}RateSPi{i}_{t}"
                for rate in ["birth", "death"]
                for i in range(len(CHANGE_TIMES) + 1)
                for t in TYPES
            ],
            job_ids=job_ids[model],
        )
        summaries.to_csv(summaries_dir / f"{model}.csv", index=False)

        weights = read_weights_dir(logs_dir)
        joblib.dump(weights, summaries_dir / f"{model}.weights.pkl")

    reference_logs_dir = base_logs_dir / BELLA_REFERENCE_MODEL
    mcc_trees_dir = summaries_dir / "mcc_trees"
    os.makedirs(mcc_trees_dir, exist_ok=True)

    for tree_file in tqdm(glob(str(reference_logs_dir / "*.trees"))):
        subprocess.run(
            [
                "treeannotator",
                tree_file,
                str(mcc_trees_dir / f"{Path(tree_file).stem}.nexus"),
                "-height",
                "median",
            ]
        )

    options = [
        ("-log", tree_file)
        for tree_file in tqdm(glob(str(reference_logs_dir / "*.trees")))
    ]
    combined_trees_file = summaries_dir / ".trees.combined.tmp.nexus"
    subprocess.run(
        ["logcombiner", *list(chain(*options)), "-o", str(combined_trees_file)]
    )
    subprocess.run(
        [
            "treeannotator",
            str(combined_trees_file),
            str(summaries_dir / "mcc.nexus"),
            "-burnin",
            "0",
            "-height",
            "median",
        ]
    )

    os.remove(combined_trees_file)

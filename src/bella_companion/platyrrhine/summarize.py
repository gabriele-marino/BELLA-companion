import json
import os
import subprocess
from itertools import chain

import joblib
from tqdm import tqdm

from bella_companion.backend.beast import read_weights_dir, summarize_logs_dir
from bella_companion.platyrrhine.settings import N_TIME_BINS, TYPES
from bella_companion.settings import settings
from bella_companion.typings import ModelJobBatch


def summarize():
    base_output_dir = settings.beast_output_dir / "platyrrhine"
    base_summaries_dir = settings.summaries_dir / "platyrrhine"

    with open(settings.job_registry_dir / "platyrrhine.json", "r") as f:
        job_ids: ModelJobBatch = json.load(f)

    for model in settings.bella_model_configs:
        logs_dir = base_output_dir / model
        summaries_dir = base_summaries_dir / model
        mcc_trees_dir = summaries_dir / "mcc_trees"
        os.makedirs(mcc_trees_dir, exist_ok=True)

        summaries = summarize_logs_dir(
            logs_dir=logs_dir,
            target_columns=[
                f"{rate}RateSPi{i}_{t}"
                for rate in ["birth", "death"]
                for i in range(N_TIME_BINS)
                for t in TYPES
            ],
            job_ids=job_ids[model],
        )
        summaries.to_csv(summaries_dir / "summaries.csv", index=False)

        weights = read_weights_dir(logs_dir)
        joblib.dump(weights, summaries_dir / "weights.pkl")

        for tree_file in tqdm(logs_dir.glob("*.trees")):
            subprocess.run(
                [
                    "treeannotator",
                    str(tree_file),
                    str(mcc_trees_dir / f"{tree_file.stem}.nexus"),
                    "-height",
                    "median",
                ]
            )

        options = [("-log", tree_file) for tree_file in tqdm(logs_dir.glob("*.trees"))]
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

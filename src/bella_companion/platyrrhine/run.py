import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from phylogenie import load_newick
from tqdm import tqdm

from bella_companion.backend import submit_beast_job
from bella_companion.platyrrhine.settings import (
    BEAST_CONFIG_PATH,
    CHANGE_TIMES,
    CHANGE_TIMES_FILE,
    TRAITS_FILE,
    TREE_FILE,
    TYPES,
)
from bella_companion.settings import BELLA_SETTINGS

JOB_IDS_FILENAME = "platyrrhine-job-ids.json"


def run_platyrrhine():
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    trees = load_newick(TREE_FILE)
    time_bins = [0, *CHANGE_TIMES]
    n_time_bins = len(time_bins)

    time_predictor = " ".join(list(map(str, np.repeat(time_bins, len(TYPES)))))
    log10BM_predictor = " ".join(map(str, TYPES * n_time_bins))

    job_ids: dict[str, dict[int, str]] = defaultdict(dict)
    for model, settings in BELLA_SETTINGS.items():
        base_log_dir = Path(os.environ["BELLA_SBATCH_LOG_DIR"]) / "platyrrhine" / model
        output_dir = base_output_dir / "platyrrhine" / model
        os.makedirs(output_dir, exist_ok=True)

        for i, tree in tqdm(
            enumerate(trees),
            desc=f"Submitting BEAST jobs for platyrrhine datasets (model: {model})",
            total=len(trees),
        ):
            data = {
                "types": ",".join(map(str, TYPES)),
                "startTypePriorProbs": "0.25 0.25 0.25 0.25",
                "birthRateUpper": "5",
                "deathRateUpper": "5",
                "samplingChangeTimes": "2.58 5.333 23.03",
                "samplingRateUpper": "5",
                "samplingRateInit": "2.5 2.5 2.5 2.5",
                "migrationRateUpper": "5",
                "migrationRateInit": "2.5 0 0 2.5 2.5 0 0 2.5 2.5 0 0 2.5",
                "treeFile": str(TREE_FILE),
                "treeIndex": str(i),
                "processLength": str(tree.origin),
                "changeTimesFile": str(CHANGE_TIMES_FILE),
                "traitsFile": str(TRAITS_FILE),
                "traitValueCol": "3",
                "timePredictor": time_predictor,
                "log10BMPredictor": log10BM_predictor,
                **settings.get_beast_data(),
            }
            job_ids[model][i] = submit_beast_job(
                data=data,
                prefix=f"{output_dir}{os.sep}",
                config_path=BEAST_CONFIG_PATH,
                log_dir=base_log_dir / str(i),
                mem_per_cpu=12000,
            )

    with open(base_output_dir / JOB_IDS_FILENAME, "w") as f:
        json.dump(job_ids, f)

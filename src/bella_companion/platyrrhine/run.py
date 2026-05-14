import json
import os

import numpy as np
from phylogenie import load_newick
from tqdm import tqdm

from bella_companion.backend.beast import submit_job
from bella_companion.platyrrhine.settings import (
    BEAST_CONFIG_PATH,
    BELLA_CONFIGS,
    CHANGE_TIMES_FILE,
    N_TIME_BINS,
    RATE_UPPER,
    TIME_BINS,
    TRAITS_FILE,
    TREE_FILE,
    TYPES,
)
from bella_companion.settings import settings
from bella_companion.typings import JobBatch


def run():
    trees = load_newick(TREE_FILE)

    time_predictor = " ".join(list(map(str, np.repeat(TIME_BINS, len(TYPES)))))
    log10BM_predictor = " ".join(map(str, TYPES * N_TIME_BINS))

    job_ids: JobBatch = {}

    base_log_dir = settings.sbatch_log_dir / "platyrrhine"
    output_dir = settings.beast_output_dir / "platyrrhine"
    os.makedirs(output_dir, exist_ok=True)

    rate_init = str(RATE_UPPER / 2)
    for i in tqdm(
        range(len(trees)), desc="Submitting BEAST jobs for platyrrhine datasets"
    ):
        data = {
            "types": ",".join(map(str, TYPES)),
            "startTypePriorProbs": "0.25 0.25 0.25 0.25",
            "birthRateUpper": RATE_UPPER,
            "deathRateUpper": RATE_UPPER,
            "samplingChangeTimes": "2.58 5.333 23.03",
            "samplingRateUpper": RATE_UPPER,
            "samplingRateInit": " ".join([rate_init] * len(TYPES)),
            "migrationRateUpper": RATE_UPPER,
            "migrationRateInit": f"{rate_init} 0 0 {rate_init} {rate_init} 0 0 {rate_init} {rate_init} 0 0 {rate_init}",
            "treeFile": str(TREE_FILE),
            "treeIndex": str(i),
            "changeTimesFile": str(CHANGE_TIMES_FILE),
            "traitsFile": str(TRAITS_FILE),
            "traitValueCol": "3",
            "timePredictor": time_predictor,
            "log10BMPredictor": log10BM_predictor,
            **BELLA_CONFIGS.get_beast_data(),
        }
        job_ids[str(i)] = submit_job(
            data=data,
            prefix=f"{output_dir}{os.sep}",
            config_path=BEAST_CONFIG_PATH,
            log_dir=base_log_dir / str(i),
            mem_per_cpu=12000,
        )

    with open(settings.job_registry_dir / "platyrrhine.json", "w") as f:
        json.dump(job_ids, f)

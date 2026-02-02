import os
from pathlib import Path

import numpy as np
from phylogenie import TreeNode, load_newick
from tqdm import tqdm

from bella_companion.backend import submit_beast_job
from bella_companion.platyrrhine.settings import (
    CHANGE_TIMES,
    CHANGE_TIMES_FILE,
    TRAITS_FILE,
    TREE_FILE,
    TYPES,
)


def run_platyrrhine():
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    output_dir = base_output_dir / "platyrrhine"
    os.makedirs(output_dir, exist_ok=True)

    trees: list[TreeNode] = load_newick(TREE_FILE)  # pyright: ignore
    time_bins = [0, *CHANGE_TIMES]
    n_time_bins = len(time_bins)

    time_predictor = " ".join(list(map(str, np.repeat(time_bins, len(TYPES)))))
    log10BM_predictor = " ".join(map(str, TYPES * n_time_bins))

    for i, tree in enumerate(
        tqdm(trees, desc="Submitting BEAST jobs for platyrrhine datasets")
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
            "layersRange": "0,1,2",
            "nodes": "16 8",
            "treeFile": str(TREE_FILE),
            "treeIndex": str(i),
            "changeTimesFile": str(CHANGE_TIMES_FILE),
            "traitsFile": str(TRAITS_FILE),
            "traitValueCol": "3",
            "processLength": str(tree.age),
            "timePredictor": time_predictor,
            "log10BMPredictor": log10BM_predictor,
        }
        submit_beast_job(
            data=data,
            prefix=f"{output_dir}{os.sep}",
            config_path=Path(__file__).parent / "beast_config.xml",
            log_dir=Path(os.environ["BELLA_SBATCH_LOG_DIR"]) / "platyrrhine" / str(i),
            mem_per_cpu=12000,
        )

import os
from pathlib import Path

import numpy as np
import pandas as pd
from phylogenie import Tree, get_node_depths, load_newick
from tqdm import tqdm

from bella_companion.backend import submit_job


def run():
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    output_dir = base_output_dir / "platyrrhine"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = Path(__file__).parent / "data"
    tree_file = data_dir / "trees.nwk"
    change_times_file = data_dir / "change_times.csv"

    trees: list[Tree] = load_newick(str(tree_file))  # pyright: ignore

    types = ["0", "1", "2", "3"]
    change_times = pd.read_csv(change_times_file, header=None).values.flatten()  # pyright: ignore
    time_bins = [0, *change_times]
    n_time_bins = len(time_bins)

    time_predictor = " ".join(list(map(str, np.repeat(time_bins, len(types)))))
    log10BM_predictor = " ".join(types * n_time_bins)

    for i, tree in enumerate(
        tqdm(trees, desc="Submitting BEAST jobs for platyrrhine datasets")
    ):
        process_length = max(get_node_depths(tree).values())
        command = " ".join(
            [
                os.environ["BELLA_RUN_BEAST_CMD"],
                "-seed 42",
                f'-D types="{",".join(types)}"',
                '-D startTypePriorProbs="0.25 0.25 0.25 0.25"',
                "-D birthRateUpper=5",
                "-D deathRateUpper=5",
                '-D samplingChangeTimes="2.58 5.333 23.03"',
                "-D samplingRateUpper=5",
                '-D samplingRateInit="2.5 2.5 2.5 2.5"',
                "-D migrationRateUpper=5",
                '-D migrationRateInit="2.5 0 0 2.5 2.5 0 0 2.5 2.5 0 0 2.5"',
                '-D nodes="16 8"',
                '-D layersRange="0,1,2"',
                f"-D treeFile={tree_file}",
                f"-D treeIndex={i}",
                f"-D changeTimesFile={change_times_file}",
                f"-D traitsFile={data_dir / 'traits.csv'}",
                "-D traitValueCol=3",
                f"-D processLength={process_length}",
                f'-D timePredictor="{time_predictor}"',
                f'-D log10BMPredictor="{log10BM_predictor}"',
                f"-prefix {output_dir}{os.sep}",
                str(Path(__file__).parent / "beast_config.xml"),
            ]
        )
        submit_job(
            command,
            Path(os.environ["BELLA_SBATCH_LOG_DIR"]) / "platyrrhine" / str(i),
            mem_per_cpu=12000,
        )

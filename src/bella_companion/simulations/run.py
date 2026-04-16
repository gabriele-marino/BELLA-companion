import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from numpy.random import default_rng
from phylogenie import load_newick
from tqdm import tqdm

from bella_companion.backend import submit_beast_job
from bella_companion.settings import BELLA_SETTINGS
from bella_companion.simulations.scenarios import SCENARIOS

JOB_IDS_FILENAME = "sim-job-ids.json"


def run_simulations():
    rng = default_rng(42)
    base_data_dir = Path(os.environ["BELLA_SIMULATIONS_DATA_DIR"])
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    base_log_dir = Path(os.environ["BELLA_SBATCH_LOG_DIR"])
    beast_configs_dir = Path(__file__).parent / "beast_configs"

    job_ids: dict[str, dict[str, dict[str, str]]] = {}
    for scenario_id, scenario in SCENARIOS.items():
        job_ids[scenario_id] = defaultdict(dict)
        data_dir = base_data_dir / scenario_id
        inference_configs_dir = beast_configs_dir / scenario.beast_configs
        log_dir = base_log_dir / scenario_id
        for tree_file in tqdm(
            glob(str(data_dir / "*.nwk")),
            desc=f"Submitting BEAST2 jobs for {scenario_id}",
        ):
            tree_id = Path(tree_file).stem
            for model in ["PA", "GLM", *BELLA_SETTINGS.keys()]:
                output_dir = base_output_dir / scenario_id / model
                os.makedirs(output_dir, exist_ok=True)

                data = scenario.beast_args | {
                    "treeFile": tree_file,
                    "treeID": tree_id,
                }
                if scenario.get_random_predictor is not None:
                    data["randomPredictor"] = " ".join(
                        map(str, scenario.get_random_predictor(rng))
                    )

                if scenario.tree_beast_args is not None:
                    (tree,) = load_newick(tree_file)
                    for arg_name, arg_func in scenario.tree_beast_args.items():
                        data[arg_name] = arg_func(tree)

                if model in BELLA_SETTINGS:
                    data.update(BELLA_SETTINGS[model].get_beast_data())

                config_filename = "BELLA" if model in BELLA_SETTINGS else model
                job_ids[scenario_id][model][tree_id] = submit_beast_job(
                    data=data,
                    prefix=f"{output_dir}{os.sep}",
                    config_path=inference_configs_dir / f"{config_filename}.xml",
                    log_dir=log_dir / model / tree_id,
                )

    with open(base_output_dir / JOB_IDS_FILENAME, "w") as f:
        json.dump(job_ids, f)

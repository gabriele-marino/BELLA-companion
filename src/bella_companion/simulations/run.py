import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from numpy.random import default_rng
from phylogenie import load_newick
from tqdm import tqdm

from bella_companion.backend.beast import submit_job
from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS
from bella_companion.typings import ModelJobBatch


def run():
    rng = default_rng(42)
    beast_configs_dir = Path(__file__).parent / "beast_configs"
    os.makedirs(settings.job_registry_dir, exist_ok=True)

    for scenario_id, scenario in SCENARIOS.items():
        data_dir = settings.simulations_data_dir / scenario_id
        inference_configs_dir = beast_configs_dir / scenario.name
        job_ids: ModelJobBatch = defaultdict(dict)
        log_dir = settings.sbatch_log_dir / scenario_id
        if log_dir.exists():
            print(
                f"Log directory {log_dir} already exists. Skipping scenario {scenario_id}."
            )
            continue

        for tree_file in tqdm(
            glob(str(data_dir / "*.nwk")),
            desc=f"Submitting BEAST2 jobs for {scenario_id}",
        ):
            tree_id = Path(tree_file).stem
            for model in ["PA", "GLM", *settings.bella_model_configs.keys()]:
                output_dir = settings.beast_output_dir / scenario_id / model
                os.makedirs(output_dir, exist_ok=True)

                data = dict(scenario.beast_static_data) | {"treeFile": tree_file}
                if scenario.beast_sample_data is not None:
                    for name, func in scenario.beast_sample_data.items():
                        data[name] = func(rng)

                if scenario.beast_tree_data is not None:
                    (tree,) = load_newick(tree_file)
                    for name, func in scenario.beast_tree_data.items():
                        data[name] = func(tree)

                if model in settings.bella_model_configs:
                    data |= settings.bella_model_configs[model].get_beast_data()

                config_filename = (
                    "BELLA" if model in settings.bella_model_configs else model
                )
                job_ids[model][tree_id] = submit_job(
                    data=data,
                    prefix=f"{output_dir}{os.sep}",
                    config_path=inference_configs_dir / f"{config_filename}.xml",
                    log_dir=log_dir / model / tree_id,
                )

        with open(settings.job_registry_dir / f"{scenario_id}.json", "w") as f:
            json.dump(job_ids, f)

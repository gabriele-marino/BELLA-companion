import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from numpy.random import default_rng
from phylogenie import Tree, load_newick
from tqdm import tqdm

from bella_companion.backend import submit_job
from bella_companion.simulations.scenarios import SCENARIOS, ScenarioType

JOB_IDS_FILENAME = "sim-job-ids.json"


def run_simulations():
    rng = default_rng(42)
    base_data_dir = Path(os.environ["BELLA_SIMULATIONS_DATA_DIR"])
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    base_log_dir = Path(os.environ["BELLA_SBATCH_LOG_DIR"])
    beast_configs_dir = Path(__file__).parent / "beast_configs"

    job_ids = {}
    for scenario_id, scenario in SCENARIOS.items():
        job_ids[scenario_id] = defaultdict(dict)
        data_dir = base_data_dir / scenario_id
        scenario_name = scenario_id.split("_")[0] if "_" in scenario_id else scenario_id
        inference_configs_dir = beast_configs_dir / scenario_name
        log_dir = base_log_dir / scenario_id
        for tree_file in tqdm(
            glob(str(data_dir / "*.nwk")),
            desc=f"Submitting BEAST2 jobs for {scenario_id}",
        ):
            tree_id = Path(tree_file).stem
            for model in ["PA", "GLM"] + [
                f"BELLA-{hidden_nodes}" for hidden_nodes in ["3_2", "16_8", "32_16"]
            ]:
                output_dir = base_output_dir / scenario_id / model
                os.makedirs(output_dir, exist_ok=True)

                beast_args = [
                    f"-D treeFile={tree_file},treeID={tree_id}",
                    f"-prefix {output_dir}{os.sep}",
                    f'-D randomPredictor="{" ".join(map(str, scenario.get_random_predictor(rng)))}"',
                ]
                beast_args.extend(
                    [
                        f'-D {key}="{value}"'
                        for key, value in scenario.beast_args.items()
                    ]
                )
                if scenario.type == ScenarioType.EPI:
                    tree: Tree = load_newick(tree_file)  # pyright: ignore
                    lastSampleTime = tree.height + tree.branch_length_or_raise()
                    beast_args.append(f"-D lastSampleTime={lastSampleTime}")

                base_command = [
                    os.environ["BELLA_RUN_BEAST_CMD"],
                    "-seed 42",
                    *beast_args,
                ]
                if model in ["PA", "GLM"]:
                    command = " ".join(
                        [*base_command, str(inference_configs_dir / f"{model}.xml")]
                    )
                else:
                    nodes = model.split("-")[1].split("_")
                    command = " ".join(
                        [
                            *base_command,
                            f'-D nodes="{" ".join(map(str, nodes))}"',
                            f'-D layersRange="{",".join(map(str, range(len(nodes) + 1)))}"',
                            str(inference_configs_dir / "BELLA.xml"),
                        ]
                    )

                job_ids[scenario_id][model][tree_id] = submit_job(
                    command, log_dir / model / tree_id
                )

    with open(base_output_dir / JOB_IDS_FILENAME, "w") as f:
        json.dump(job_ids, f)

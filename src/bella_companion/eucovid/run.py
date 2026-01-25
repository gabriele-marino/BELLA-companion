import os
from itertools import product
from pathlib import Path

from bella_companion.backend import submit_beast_job
from bella_companion.eucovid.settings import DATA_DIR, MSA_FILE


def run_eucovid():
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "eucovid"
    base_log_dir = Path(os.environ["BELLA_SBATCH_LOG_DIR"]) / "eucovid"
    beast_configs_dir = Path(__file__).parent / "beast_configs"

    for seed, (model, experiment, predictors) in product(
        range(1, 4),
        [
            ("GLM", "flights_over_populations", ["flights_over_populations"]),
            ("BELLA", "flights_over_populations", ["flights_over_populations"]),
            ("BELLA", "flights_and_populations", ["flights", "populations"]),
        ],
    ):
        output_dir = base_output_dir / experiment / model / str(seed)
        log_dir = base_log_dir / experiment / model / str(seed)
        predictors_dir = DATA_DIR / experiment
        data = {
            "msa_file": str(MSA_FILE),
            "changeTimesFile": str(predictors_dir / "changetimes.csv"),
            "predictorFiles": ",".join(
                [str(predictors_dir / f"{predictor}.csv") for predictor in predictors]
            ),
        }
        if model == "BELLA":
            data["layersRange"] = "0,1,2"
            data["nodes"] = "16 8"

        os.makedirs(output_dir, exist_ok=True)
        submit_beast_job(
            data=data,
            prefix=f"{output_dir}{os.sep}",
            config_path=beast_configs_dir / f"{model}.xml",
            log_dir=log_dir,
            time="240:00:00",
            cpus=128,
            mem_per_cpu=12000,
            seed=seed,
        )

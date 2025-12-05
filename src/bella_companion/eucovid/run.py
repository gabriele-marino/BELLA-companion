import os
from itertools import product
from pathlib import Path

from bella_companion.backend import submit_job


def run():
    base_output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "eucovid"
    base_log_dir = Path(os.environ["BELLA_SBATCH_LOG_DIR"]) / "eucovid"
    data_dir = Path(__file__).parent / "data"
    beast_configs_dir = Path(__file__).parent / "beast_configs"

    for seed, (model, experiment, predictors) in product(
        range(1, 4),
        [
            ("GLM", "flights_over_population", ["flights_over_population"]),
            ("BELLA", "flights_over_population", ["flights_over_population"]),
            ("BELLA", "flights_and_populations", ["flights", "populations"]),
        ],
    ):
        output_dir = base_output_dir / experiment / model / str(seed)
        log_dir = base_log_dir / experiment / model / str(seed)
        predictors_dir = data_dir / experiment
        extra_data = "-D predictorFiles=" + ",".join(
            [str(predictors_dir / f"{predictor}.csv") for predictor in predictors]
        )
        if model == "BELLA":
            extra_data += ' -D layersRange="0,1,2",nodes="16 8"'

        os.makedirs(output_dir, exist_ok=True)
        submit_job(
            " ".join(
                [
                    os.environ["BELLA_RUN_BEAST_CMD"],
                    f"-seed {seed}",
                    f"-D msa_file={data_dir / 'msa.fasta'}",
                    f"-D changeTimesFile={predictors_dir / 'changetimes.csv'}",
                    extra_data,
                    f"-prefix {output_dir}{os.sep}",
                    str(beast_configs_dir / f"{model}.xml"),
                ]
            ),
            log_dir,
            cpus=128,
            mem_per_cpu=12000,
        )

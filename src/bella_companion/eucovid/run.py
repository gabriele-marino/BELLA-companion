import os
from itertools import product
from pathlib import Path

from phylogenie import load_fasta

from bella_companion.backend.beast import submit_job
from bella_companion.eucovid.settings import MSA_FILE, N_SEEDS
from bella_companion.settings import settings

THIS_DIR = Path(__file__).parent


def run_eucovid():
    base_output_dir = settings.bella_beast_output_dir / "eucovid"
    base_log_dir = settings.bella_sbatch_log_dir / "eucovid"
    beast_configs_dir = Path(__file__).parent / "beast_configs"

    for seed, (model, experiment, predictors) in product(
        range(1, N_SEEDS + 1),
        [
            ("BELLA", "flights_and_populations", ["flights", "populations"]),
            ("BELLA", "flights_over_populations", ["flights_over_populations"]),
            ("GLM", "flights_over_populations", ["flights_over_populations"]),
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

        data["ReInitChinaAfterLockdown"] = "1.0"
        data["ReInitFrance"] = "1.1"
        data["ReInitGermany"] = "1.2"
        data["ReInitItaly"] = "1.3"
        data["ReInitOtherEU"] = "1.4"
        data["ReInitChinaBeforeLockdown"] = "1.01"

        data["samplingProportionInitChina"] = "1.1E-4"
        data["samplingProportionInitFrance"] = "1.2E-3"
        data["samplingProportionInitGermany"] = "1.3E-2"
        data["samplingProportionInitItaly"] = "1.4E-3"
        data["samplingProportionInitOtherEU"] = "1.5E-3"

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

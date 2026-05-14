import json
import os
from collections import defaultdict
from itertools import product

from phylogenie.io import load_fasta

from bella_companion.backend.beast import submit_job
from bella_companion.eucovid.settings import (
    BEAST_CONFIGS_DIR,
    BELLA_CONFIGS,
    COUNTRIES,
    MSA_FILE,
    N_CASES,
    N_SEEDS,
    PREDICTOR_TRANSFORMED,
)
from bella_companion.settings import settings
from bella_companion.typings import ModelJobBatch


def run():
    base_output_dir = settings.beast_output_dir / "eucovid"
    base_log_dir = settings.sbatch_log_dir / "eucovid"

    msa = load_fasta(MSA_FILE)
    sequences_per_country: dict[str, int] = defaultdict(int)
    for seq in msa:
        _, country, _ = seq.id.split("|")
        sequences_per_country[country] += 1

    job_ids: ModelJobBatch = defaultdict(dict)
    for seed, model in product(range(1, N_SEEDS + 1), ["GLM", "BELLA"]):
        output_dir = base_output_dir / model / str(seed)
        log_dir = base_log_dir / model / str(seed)
        data = {
            "msa_file": str(MSA_FILE),
            "predictor": " ".join(map(str, PREDICTOR_TRANSFORMED.flatten().tolist())),
            **BELLA_CONFIGS.get_beast_data(),
            "ReInitChinaAfterLockdown": "1.0",
            "ReInitFrance": "1.1",
            "ReInitGermany": "1.2",
            "ReInitItaly": "1.3",
            "ReInitOtherEU": "1.4",
            "ReInitChinaBeforeLockdown": "1.01",
        }
        for country, n_cases in zip(COUNTRIES, N_CASES):
            n_sequences = sequences_per_country[country]
            data[f"samplingProportion{country}LowerBound"] = n_sequences / n_cases / 10
            data[f"samplingProportion{country}UpperBound"] = n_sequences / n_cases
            data[f"samplingProportionInit{country}"] = n_sequences / n_cases / 5

        os.makedirs(output_dir, exist_ok=True)
        job_ids[model][str(seed)] = submit_job(
            data=data,
            prefix=f"{output_dir}{os.sep}",
            config_path=BEAST_CONFIGS_DIR / f"{model}.xml",
            log_dir=log_dir,
            time="240:00:00",
            cpus=128,
            mem_per_cpu=12000,
            seed=seed,
        )

    with open(settings.job_registry_dir / "eucovid.json", "w") as f:
        json.dump(job_ids, f)

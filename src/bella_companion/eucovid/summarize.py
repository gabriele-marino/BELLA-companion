import json
import os
import subprocess
from itertools import chain

from bella_companion.eucovid.settings import N_SEEDS
from bella_companion.settings import settings
from bella_companion.typings import ModelJobBatch


def summarize():
    logs_dir = settings.beast_output_dir / "eucovid"
    base_summaries_dir = settings.summaries_dir / "eucovid"
    with open(settings.job_registry_dir / "eucovid.json", "r") as f:
        job_ids: ModelJobBatch = json.load(f)

    for model in job_ids:
        log_dir = logs_dir / model
        summaries_dir = base_summaries_dir / model
        os.makedirs(summaries_dir, exist_ok=True)

        options = [
            ("-log", str(log_dir / str(seed) / "MCMC.log"))
            for seed in range(1, N_SEEDS + 1)
        ]
        subprocess.run(
            [
                "logcombiner",
                *list(chain(*options)),
                "-o",
                str(summaries_dir / "MCMC.combined.log"),
            ]
        )

        options = [
            ("-log", str(log_dir / str(seed) / "typedNodeTrees.trees"))
            for seed in range(1, N_SEEDS + 1)
        ]
        combined_trees_file = summaries_dir / ".trees.combined.tmp.nwk"
        subprocess.run(
            ["logcombiner", *list(chain(*options)), "-o", str(combined_trees_file)]
        )
        subprocess.run(
            [
                "treeannotator",
                "-file",
                str(combined_trees_file),
                str(summaries_dir / "mcc.nexus"),
                "-burnin",
                "0",
            ]
        )
        os.remove(combined_trees_file)

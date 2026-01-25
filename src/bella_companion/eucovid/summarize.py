import os
import subprocess
from itertools import chain
from pathlib import Path


def summarize_eucovid():
    logs_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"]) / "eucovid"
    base_summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"], "eucovid")

    for model, experiment in [
        ("BELLA", "flights_and_populations"),
        ("BELLA", "flights_over_populations"),
        ("GLM", "flights_over_populations"),
    ]:
        log_dir = logs_dir / experiment / model
        summaries_dir = base_summaries_dir / experiment / model
        os.makedirs(summaries_dir, exist_ok=True)

        options = [
            ("-log", str(log_dir / str(seed) / "MCMC.log")) for seed in range(1, 4)
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
            for seed in range(1, 4)
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

import json
import os
from pathlib import Path

import joblib

from bella_companion.backend import read_weights_dir, summarize_logs_dir
from bella_companion.simulations.run import JOB_IDS_FILENAME
from bella_companion.simulations.scenarios import SCENARIOS


def summarize_simulations():
    output_dir = Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
    with open(output_dir / JOB_IDS_FILENAME, "r") as f:
        job_ids: dict[str, dict[str, dict[str, str]]] = json.load(f)

    for scenario_name, scenario in SCENARIOS.items():
        summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / scenario_name
        os.makedirs(summaries_dir, exist_ok=True)
        for model in job_ids[scenario_name]:
            logs_dir = output_dir / scenario_name / model
            print(f"Summarizing {scenario_name} - {model}")
            summaries = summarize_logs_dir(
                logs_dir,
                target_columns=[c for t in scenario.targets.values() for c in t],
                job_ids=job_ids[scenario_name][model],
            )
            summaries.to_csv(summaries_dir / f"{model}.csv", index=False)
            if model.startswith("BELLA"):
                weights = read_weights_dir(logs_dir)
                joblib.dump(weights, summaries_dir / f"{model}.weights.pkl")

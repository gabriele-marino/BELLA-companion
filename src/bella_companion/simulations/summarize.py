import json
import os

import joblib

from bella_companion.backend.beast import read_weights_dir, summarize_logs_dir
from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS
from bella_companion.typings import ModelJobBatch


def summarize():
    output_dir = settings.beast_output_dir

    for scenario_id, scenario in SCENARIOS.items():
        with open(settings.job_registry_dir / f"{scenario_id}.json", "r") as f:
            model_job_ids: ModelJobBatch = json.load(f)

        summaries_dir = settings.summaries_dir / scenario_id
        os.makedirs(summaries_dir, exist_ok=True)
        for model, job_ids in model_job_ids.items():
            logs_dir = output_dir / scenario_id / model
            print(f"Summarizing {scenario_id} - {model}")
            summaries = summarize_logs_dir(
                logs_dir,
                target_columns=[
                    target_key
                    for target in scenario.targets
                    for target_key in target.keys
                ],
                job_ids=job_ids,
            )
            summaries.to_csv(summaries_dir / f"{model}.csv", index=False)
            if model in settings.bella_model_configs:
                weights = read_weights_dir(logs_dir)
                joblib.dump(weights, summaries_dir / f"{model}.weights.pkl")

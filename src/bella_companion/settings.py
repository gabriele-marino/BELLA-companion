from collections.abc import Mapping
from pathlib import Path

import yaml
from joblib import Memory
from pydantic import BaseModel

from bella_companion.typings import BELLAConfig, ModelID, ScenarioID


class Settings(BaseModel):
    simulations_data_dir: Path

    sbatch_log_dir: Path
    beast_output_dir: Path
    job_registry_dir: Path

    bella_model_configs: Mapping[ModelID, BELLAConfig]

    summaries_dir: Path

    tables_dir: Path

    figures_dir: Path
    bella_reference_models: Mapping[ScenarioID, ModelID]
    model_colors: Mapping[ModelID, str]


SETTINGS_FILE = "settings.yaml"
with open(SETTINGS_FILE) as f:
    settings = Settings.model_validate(yaml.safe_load(f))

memory = Memory(location=".bella-cache", verbose=0)

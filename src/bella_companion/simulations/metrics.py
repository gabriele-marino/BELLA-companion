import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bella_companion.simulations.scenarios import SCENARIOS

METRICS_FILENAME = "sim-metrics.json"


def _mae(summary: pd.DataFrame, true_values: dict[str, float]) -> float:
    preds = np.array([summary[f"{target}_median"].median() for target in true_values])
    targets = np.array(list(true_values.values()))
    return np.mean(np.abs(preds - targets), dtype=float)


def _coverage(summary: pd.DataFrame, true_values: dict[str, float]) -> float:
    coverages = [
        (
            (summary[f"{target}_lower"] <= true_values[target])
            & (true_values[target] <= summary[f"{target}_upper"])
        ).sum()
        / len(summary)
        for target in true_values
    ]
    return float(np.mean(coverages))


def _avg_ci_width(summary: pd.DataFrame, true_values: dict[str, float]) -> float:
    widths = [
        np.median(summary[f"{target}_upper"] - summary[f"{target}_lower"])
        for target in true_values
    ]
    return float(np.mean(widths))


def _mean_ess_per_hour(summary: pd.DataFrame, targets: list[str]) -> float:
    mean_ess = pl.mean_horizontal([f"{t}_ess" for t in targets])
    mean_ess_per_hour = mean_ess / pl.col("total_hours")
    return summary.select(mean_ess_per_hour).mean().item()


def metrics():
    base_summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"])
    estimates = {}
    mean_ess_per_hour = {}
    for name, scenario in SCENARIOS.items():
        summaries_dir = base_summaries_dir / name
        summaries = {
            Path(log_summary).stem: pl.read_csv(log_summary)
            for log_summary in glob(str(summaries_dir / "*.csv"))
        }
        estimates[name] = {
            target: {
                model: {
                    "MAE": _mae(summary, true_values),
                    "coverage": _coverage(summary, true_values),
                    "avg_CI_width": _avg_ci_width(summary, true_values),
                }
                for model, summary in summaries.items()
            }
            for target, true_values in scenario.targets.items()
        }
        mean_ess_per_hour[name] = {
            model: _mean_ess_per_hour(
                summary, [target for v in scenario.targets.values() for target in v]
            )
            for model, summary in summaries.items()
        }
    with open(base_summaries_dir / METRICS_FILENAME, "w") as f:
        json.dump({"estimates": estimates, "mean_ess_per_hour": mean_ess_per_hour}, f)

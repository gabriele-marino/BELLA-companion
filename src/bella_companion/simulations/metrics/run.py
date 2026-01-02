import os
from collections.abc import Callable
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bella_companion.backend import (
    avg_ci_width_from_summaries,
    coverage_from_summaries,
    mae_from_summaries,
    mean_ess_per_hour_from_summaries,
)
from bella_companion.simulations.scenarios import SCENARIOS


def _mae(summaries: pd.DataFrame, targets: dict[str, float]) -> float:
    return mae_from_summaries(summaries, targets)


def _coverage(summaries: pd.DataFrame, targets: dict[str, float]) -> float:
    return coverage_from_summaries(summaries, targets)


def _avg_ci_width(summaries: pd.DataFrame, targets: dict[str, float]) -> float:
    return avg_ci_width_from_summaries(summaries, list(targets))


def _mean_ess_per_hour(summaries: pd.DataFrame, targets: dict[str, float]) -> float:
    return mean_ess_per_hour_from_summaries(summaries, list(targets))


def _format_results(
    results: dict[str, float], lower_is_better: bool | None = None
) -> dict[str, str]:
    formatted_results = {model: f"{value:.3f}" for model, value in results.items()}
    if lower_is_better is not None:
        sorted_models = sorted(
            formatted_results.items(),
            key=lambda item: item[1],
            reverse=not lower_is_better,
        )
        best_model, best_value = sorted_models[0]
        formatted_results[best_model] = f"\\textbf{{{best_value}}}"
        second_best_model, second_best_value = sorted_models[1]
        formatted_results[second_best_model] = f"\\underline{{{second_best_value}}}"
    return formatted_results


def _run_metric(
    metric_label: str,
    metric_func: Callable[[pd.DataFrame, dict[str, float]], float],
    metric_name: str,
    lower_is_better: bool | None = None,
) -> str:
    base_summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"])

    with open(Path(__file__).parent / "template.tex", "r") as f:
        template = f.read()

    output_table = template.replace("{{METRIC_NAME}}", metric_name)
    output_table = output_table.replace("{{METRIC_LABEL}}", metric_label)
    output_table = output_table.replace(
        "{{CAPTION_EXTRA}}",
        ""
        if lower_is_better is None
        else "Bold indicates the best, underlined indicates the second-best.",
    )

    for scenario_name, scenario in SCENARIOS.items():
        summaries_dir = base_summaries_dir / scenario_name
        models_summaries = {
            Path(summary).stem: pd.read_csv(summary)  # pyright: ignore
            for summary in glob(str(summaries_dir / "*.csv"))
        }

        results = {
            target: {
                model: metric_func(summaries, scenario.targets[target])
                for model, summaries in models_summaries.items()
            }
            for target in scenario.targets
        }

        for target in scenario.targets:
            formatted_results = _format_results(results[target], lower_is_better)
            for model, value in formatted_results.items():
                placeholder = f"{{{{{scenario_name}-{model}-{target}}}}}"
                output_table = output_table.replace(placeholder, str(value))

        if len(scenario.targets) > 1:
            mean_results = {
                model: np.mean(
                    [results[target][model] for target in scenario.targets], dtype=float
                )
                for model in models_summaries
            }
            formatted_mean_results = _format_results(mean_results, lower_is_better)
            for model, value in formatted_mean_results.items():
                placeholder = f"{{{{{scenario_name}-{model}-average}}}}"
                output_table = output_table.replace(placeholder, str(value))

    return output_table


def run_metrics():
    outputs_dir = Path(os.environ["BELLA_TABLES_DIR"])
    os.makedirs(outputs_dir, exist_ok=True)

    with open(outputs_dir / "mae.tex", "w") as f:
        f.write(
            _run_metric(
                metric_label="mae",
                metric_func=_mae,
                metric_name="Mean absolute error (MAE)",
                lower_is_better=True,
            )
        )

    with open(outputs_dir / "coverage.tex", "w") as f:
        f.write(
            _run_metric(
                metric_label="coverage", metric_func=_coverage, metric_name="Coverage"
            )
        )

    with open(outputs_dir / "avg_ci_width.tex", "w") as f:
        f.write(
            _run_metric(
                metric_label="avg_ci_width",
                metric_func=_avg_ci_width,
                metric_name="Average 95\\% credible interval width",
            )
        )

    with open(outputs_dir / "mean_ess_per_hour.tex", "w") as f:
        f.write(
            _run_metric(
                metric_label="mean_ess_per_hour",
                metric_func=_mean_ess_per_hour,
                metric_name="Mean ESS per hour",
            )
        )

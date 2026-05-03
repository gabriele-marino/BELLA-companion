import os
from glob import glob
from pathlib import Path

import pandas as pd

from bella_companion.metrics import (
    CoefficientOfVariation,
    Coverage,
    MeanESSPerHour,
    Metric,
    NormalizedMAE,
)
from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS
from bella_companion.typings import ModelID


def _format_results(
    results: dict[ModelID, float],
    lower_is_better: bool,
    format_best_values: bool,
    n_decimals: int | None,
) -> dict[ModelID, float | str]:
    formatted_results: dict[ModelID, float | str] = {
        model: round(value, n_decimals) for model, value in results.items()
    }
    if format_best_values:
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


def _save_metric_table(
    metric: Metric,
    n_decimals: int | None = 3,
    format_best_values: bool = False,
):
    base_summaries_dir = settings.summaries_dir

    with open(Path(__file__).parent / "template.tex", "r") as f:
        template = f.read()

    output_table = template.replace("{{METRIC_NAME}}", metric.name)
    output_table = output_table.replace("{{METRIC_LABEL}}", metric.id)
    output_table = output_table.replace(
        "{{CAPTION_EXTRA}}",
        ""
        if format_best_values
        else "Bold indicates the best, underlined indicates the second-best.",
    )

    for scenario_id, scenario in SCENARIOS.items():
        summaries_dir = base_summaries_dir / scenario_id
        models_summaries = {
            Path(summary).stem: pd.read_csv(summary)  # pyright: ignore
            for summary in glob(str(summaries_dir / "*.csv"))
        }

        model_results = {
            model: metric.aggregate(summaries, scenario.targets)
            for model, summaries in models_summaries.items()
        }

        formatted_results = _format_results(
            model_results, metric.lower_is_better, format_best_values, n_decimals
        )
        for model, value in formatted_results.items():
            placeholder = f"{{{{{scenario_id}-{model}}}}}"
            output_table = output_table.replace(placeholder, str(value))

    with open(settings.tables_dir / f"{metric.id}.tex", "w") as f:
        f.write(output_table)


def build_tables():
    os.makedirs(settings.tables_dir, exist_ok=True)
    _save_metric_table(metric=NormalizedMAE(), format_best_values=True)
    _save_metric_table(metric=Coverage())
    _save_metric_table(metric=CoefficientOfVariation())
    _save_metric_table(metric=MeanESSPerHour(), n_decimals=None)

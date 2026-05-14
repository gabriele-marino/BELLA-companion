import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from bella_companion.metrics import MAPE, CoefficientOfVariation, Coverage
from bella_companion.settings import settings
from bella_companion.simulations.plot.common import (
    plot_per_run_metric_through_time,
    plot_per_study_metric_through_time,
)
from bella_companion.simulations.scenarios import EPI_SKYLINE_SCENARIOS


def plot_metrics_through_time(output_file: Path):
    _, axes = plt.subplots(3, 3, figsize=(10, 10), sharex="col", layout="constrained")  # pyright: ignore

    for row, label in zip(axes, string.ascii_lowercase):
        ax = row[0]
        ax.text(
            -0.2, 0.95, label, transform=ax.transAxes, fontsize=15, fontweight="bold"
        )

    reference_model = settings.bella_reference_models["epi-skyline"]
    models = ["PA", "GLM", reference_model]
    for i, (scenario_id, scenario) in enumerate(EPI_SKYLINE_SCENARIOS.items()):
        summaries_dir = settings.summaries_dir / scenario_id
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in models
        }
        (reproduction_number,) = scenario.targets
        plot_per_run_metric_through_time(
            ax=axes[i, 0],
            metric=MAPE(),
            target=reproduction_number,
            models_summaries=models_summaries,
            times_are_ages=False,
        )
        plot_per_study_metric_through_time(
            ax=axes[i, 1],
            metric=Coverage(),
            target=reproduction_number,
            models_summaries=models_summaries,
            times_are_ages=False,
        )
        plot_per_run_metric_through_time(
            ax=axes[i, 2],
            metric=CoefficientOfVariation(),
            target=reproduction_number,
            models_summaries=models_summaries,
            times_are_ages=False,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time bin")

    handle = [
        Patch(facecolor=settings.model_colors[model], edgecolor="none", label=model)
        for model in models
    ]
    axes[0, 1].legend(
        handles=handle,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.55),
        fontsize=14,
        handlelength=3,
        handleheight=1.5,
    )

    plt.savefig(output_file)  # pyright: ignore

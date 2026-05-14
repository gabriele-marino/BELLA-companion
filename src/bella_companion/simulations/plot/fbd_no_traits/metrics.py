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
from bella_companion.simulations.scenarios import FBD_NO_TRAITS_SCENARIOS


def plot_metrics_through_time(output_file: Path):
    _, axes = plt.subplots(6, 3, figsize=(12, 16), sharex="col", layout="constrained")  # pyright: ignore

    for n_row, label in zip([0, 2, 4], ["a", "b", "c"]):
        ax = axes[n_row, 0]
        ax.text(
            -0.22, 0.95, label, transform=ax.transAxes, fontsize=18, fontweight="bold"
        )
    for n_row in range(6):
        ax = axes[n_row, 0]
        ax.text(
            -0.22,
            0.5,
            r"$\mu$" if n_row % 2 else r"$\lambda$",
            transform=ax.transAxes,
            fontsize=18,
        )

    reference_model = settings.bella_reference_models["fbd-no-traits"]
    models = ["PA", "GLM", reference_model]
    for i, (scenario_id, scenario) in enumerate(FBD_NO_TRAITS_SCENARIOS.items()):
        summaries_dir = settings.summaries_dir / scenario_id
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in reversed(models)
        }

        for j, target in enumerate(scenario.targets):
            plot_per_run_metric_through_time(
                ax=axes[i * 2 + j, 0],
                metric=MAPE(),
                target=target,
                models_summaries=models_summaries,
                times_are_ages=True,
            )
            plot_per_study_metric_through_time(
                ax=axes[i * 2 + j, 1],
                metric=Coverage(),
                target=target,
                models_summaries=models_summaries,
                times_are_ages=True,
            )
            plot_per_run_metric_through_time(
                ax=axes[i * 2 + j, 2],
                metric=CoefficientOfVariation(),
                target=target,
                models_summaries=models_summaries,
                times_are_ages=True,
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time bin")
        ax.invert_xaxis()

    handle = [
        Patch(facecolor=settings.model_colors[model], edgecolor="none", label=model)
        for model in models
    ]
    axes[0, 1].legend(
        handles=handle,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.65),
        fontsize=14,
        handlelength=3,
        handleheight=1.5,
    )

    plt.savefig(output_file)  # pyright: ignore

import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from bella_companion.settings import settings
from bella_companion.simulations.plot.common import plot_medians_ribbon
from bella_companion.simulations.scenarios import EPI_SKYLINE_SCENARIOS, EPI_TIME_AXIS


def plot_ribbons(output_file: Path):
    _, axes = plt.subplots(  # pyright: ignore
        3, 3, figsize=(8, 8), sharey="row", sharex="col", layout="constrained"
    )

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
        for j, model in enumerate(models):
            plot_medians_ribbon(
                ax=axes[i, j],
                time_axis=EPI_TIME_AXIS,
                target=reproduction_number,
                summaries=models_summaries[model],
                color=settings.model_colors[model],
            )

    for ax in axes[:, 0]:
        ax.set_ylabel("$R_t$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")

    for col, model in enumerate(models):
        handle = Patch(
            facecolor=settings.model_colors[model], edgecolor="none", label=model
        )
        axes[0, col].legend(
            handles=[handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            fontsize=14,
            handlelength=3,
            handleheight=1.5,
        )

    plt.savefig(output_file)  # pyright: ignore

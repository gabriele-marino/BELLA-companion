from itertools import product
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
from bella_companion.simulations.scenarios import FBD_2TRAITS


def plot_metrics_through_time(output_file: Path):
    _, axes = plt.subplots(8, 3, figsize=(12, 20), sharex="col", layout="constrained")  # pyright: ignore

    for n_row, label in zip(range(0, 8, 4), ["a", "b"]):
        ax = axes[n_row, 0]
        ax.text(
            -0.35, 0.85, label, transform=ax.transAxes, fontsize=24, fontweight="bold"
        )
    for n_row, (label, state) in zip(
        range(8), product([r"\lambda", r"\mu"], ["00", "01", "10", "11"])
    ):
        ax = axes[n_row, 0]
        ax.text(
            -0.35,
            0.5,
            rf"${label}^" "{" f"({state})" "}$",
            transform=ax.transAxes,
            fontsize=18,
        )

    reference_model = settings.bella_reference_models["fbd-2traits"]
    models = ["PA", "GLM", reference_model]
    for i, rates in enumerate([FBD_2TRAITS.birth_rates, FBD_2TRAITS.death_rates]):
        summaries_dir = settings.summaries_dir / "fbd-2traits"
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in reversed(models)
        }

        for j, rate in enumerate(rates):
            plot_per_run_metric_through_time(
                ax=axes[i * 4 + j, 0],
                metric=MAPE(),
                target=rate,
                models_summaries=models_summaries,
                times_are_ages=True,
            )
            plot_per_study_metric_through_time(
                ax=axes[i * 4 + j, 1],
                metric=Coverage(),
                target=rate,
                models_summaries=models_summaries,
                times_are_ages=True,
            )
            plot_per_run_metric_through_time(
                ax=axes[i * 4 + j, 2],
                metric=CoefficientOfVariation(),
                target=rate,
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

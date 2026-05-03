import os
import string

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from bella_companion.settings import settings
from bella_companion.simulations.plot.common import plot_medians_ribbon
from bella_companion.simulations.scenarios import FBD_NO_TRAITS_SCENARIOS, FBD_TIME_AXIS


def plot_ribbons():
    base_output_dir = settings.figures_dir / "fbd-no-traits"
    os.makedirs(base_output_dir, exist_ok=True)

    _, axes = plt.subplots(  # pyright: ignore
        6, 3, figsize=(12, 16), sharey="row", sharex="col", layout="constrained"
    )
    for n_row, label in zip(range(0, 6, 2), string.ascii_lowercase):
        ax = axes[n_row, 0]
        ax.text(
            -0.22, 0.95, label, transform=ax.transAxes, fontsize=18, fontweight="bold"
        )

    reference_model = settings.bella_reference_models["fbd-no-traits"]
    models = ["PA", "GLM", reference_model]
    for i, (scenario_id, scenario) in enumerate(FBD_NO_TRAITS_SCENARIOS.items()):
        summaries_dir = settings.summaries_dir / scenario_id
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in models
        }
        for j, model in enumerate(models):
            for k, target in enumerate(scenario.targets):
                plot_medians_ribbon(
                    ax=axes[i * 2 + k, j],
                    time_axis=FBD_TIME_AXIS,
                    target=target,
                    summaries=models_summaries[model],
                    color=settings.model_colors[model],
                    times_are_ages=True,
                )

    for i, ax in enumerate(axes[0, :]):
        ax.set_ylabel(r"$\mu$" if i % 2 else r"$\lambda$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")
        ax.invert_xaxis()

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

    plt.savefig(base_output_dir / "ribbons.svg")  # pyright: ignore

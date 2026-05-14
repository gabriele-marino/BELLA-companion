import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.settings import settings
from bella_companion.simulations.plot.common import plot_medians_ribbon
from bella_companion.simulations.scenarios import EPI_SKYLINE_SCENARIOS, EPI_TIME_AXIS


def plot_sensitivity_ribbons(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    reference_model = settings.bella_reference_models["epi-skyline"]
    models = [m for m in settings.bella_model_configs if m != reference_model]
    for scenario_id, scenario in EPI_SKYLINE_SCENARIOS.items():
        summaries_dir = settings.summaries_dir / scenario_id

        _, axes = plt.subplots(  # pyright: ignore
            2, 4, figsize=(12, 6), sharey="row", sharex="col", layout="constrained"
        )

        for ax, model in zip(axes.flatten(), models):
            ax.text(
                0.7,
                0.9,
                model,
                transform=ax.transAxes,
                fontsize=12,
                fontfamily="monospace",
            )

            summaries = pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            (reproduction_number,) = scenario.targets
            plot_medians_ribbon(
                ax=ax,
                time_axis=EPI_TIME_AXIS,
                target=reproduction_number,
                summaries=summaries,
                color="#777777",
            )

        for ax in axes[:, 0]:
            ax.set_ylabel("$R_t$")
        for ax in axes[-1, :]:
            ax.set_xlabel("Time")

        plt.savefig(output_dir / f"{scenario_id}.pdf")  # pyright: ignore

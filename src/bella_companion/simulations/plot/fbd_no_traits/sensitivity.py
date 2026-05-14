import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.settings import settings
from bella_companion.simulations.plot.common import plot_medians_ribbon
from bella_companion.simulations.scenarios import FBD_NO_TRAITS_SCENARIOS, FBD_TIME_AXIS


def plot_sensitivity_ribbons(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    reference_model = settings.bella_reference_models["fbd-no-traits"]
    models = [m for m in settings.bella_model_configs if m != reference_model]
    for scenario_id, scenario in FBD_NO_TRAITS_SCENARIOS.items():
        summaries_dir = settings.summaries_dir / scenario_id

        _, axes = plt.subplots(  # pyright: ignore
            4, 4, figsize=(12, 12), sharey="row", sharex="col", layout="constrained"
        )

        for n_row, label in zip([0, 2], ["a", "b"]):
            ax = axes[n_row, 0]
            ax.text(
                -0.22,
                0.95,
                label,
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
            )

        for ax, ((i, target), model) in zip(
            axes.flatten(), product(enumerate(scenario.targets), models)
        ):
            ax.text(
                0.7,
                0.9,
                model,
                transform=ax.transAxes,
                fontsize=12,
                fontfamily="monospace",
            )

            summaries = pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            plot_medians_ribbon(
                ax=ax,
                time_axis=FBD_TIME_AXIS,
                target=target,
                summaries=summaries,
                color=f"C{i}",
                times_are_ages=True,
            )

        for i, ax in enumerate(axes[:, 0]):
            label = r"$\lambda$" if i < 2 else r"$\mu$"
            ax.set_ylabel(label)
        for ax in axes[-1, :]:
            ax.set_xlabel("Time")
            ax.invert_xaxis()

        plt.savefig(output_dir / f"{scenario_id}.pdf")  # pyright: ignore

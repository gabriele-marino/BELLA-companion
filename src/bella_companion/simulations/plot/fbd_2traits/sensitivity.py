from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.settings import settings
from bella_companion.simulations.plot.fbd_2traits.common import (
    plot_estimate_interquantiles,
)
from bella_companion.simulations.scenarios import FBD_2TRAITS


def plot_sensitivity_ribbons(output_file: Path):
    reference_model = settings.bella_reference_models["fbd-2traits"]
    models = [m for m in settings.bella_model_configs if m != reference_model]
    summaries_dir = settings.summaries_dir / "fbd-2traits"

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

    for ax, (targets, model) in zip(
        axes.flatten(),
        product([FBD_2TRAITS.birth_rates, FBD_2TRAITS.death_rates], models),
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
        plot_estimate_interquantiles(ax=ax, summaries=summaries, targets=targets)

    for i, ax in enumerate(axes[:, 0]):
        label = r"$\lambda$" if i < 2 else r"$\mu$"
        ax.set_ylabel(label)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")
        ax.invert_xaxis()

    plt.savefig(output_file)  # pyright: ignore

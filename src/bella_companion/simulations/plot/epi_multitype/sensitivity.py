from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.settings import settings
from bella_companion.simulations.plot.epi_multitype.common import plot_estimates


def plot_estimate_sensitivity(output_file: Path):
    reference_model = settings.bella_reference_models["epi-multitype"]
    models = [m for m in settings.bella_model_configs if m != reference_model]
    summaries_dir = settings.summaries_dir / "epi-multitype"

    _, axes = plt.subplots(  # pyright: ignore
        2, 4, figsize=(12, 6), sharey="row", sharex="col", layout="constrained"
    )

    for ax, model in zip(axes.flatten(), models):
        ax.text(
            0.05,
            0.9,
            model,
            transform=ax.transAxes,
            fontsize=12,
            fontfamily="monospace",
        )

        summaries = pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        plot_estimates(ax=ax, summaries=summaries, color="#777777")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$m_{ij}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x_{ij}$")

    plt.savefig(output_file)  # pyright: ignore

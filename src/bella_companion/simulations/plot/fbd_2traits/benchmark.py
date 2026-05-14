from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.settings import settings
from bella_companion.simulations.plot.fbd_2traits.common import (
    plot_estimate_interquantiles,
)
from bella_companion.simulations.scenarios import FBD_2TRAITS


def plot_benchmark_ribbons(output_file: Path):
    summaries_dir = settings.summaries_dir / "fbd-2traits"

    _, axes = plt.subplots(  # pyright: ignore
        2, 2, figsize=(6, 6), sharex="col", layout="constrained"
    )

    for ax, label in zip(axes[:, 0], ["a", "b"]):
        ax.text(
            -0.22, 0.95, label, transform=ax.transAxes, fontsize=18, fontweight="bold"
        )

    for i, model in enumerate(["PA", "GLM"]):
        summaries = pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for ax, target in zip(
            axes[i], [FBD_2TRAITS.birth_rates, FBD_2TRAITS.death_rates]
        ):
            plot_estimate_interquantiles(ax=ax, summaries=summaries, targets=target)

    axes[0, 0].set_ylabel(r"$\lambda$")
    axes[1, 0].set_ylabel(r"$\mu$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time")
        ax.invert_xaxis()

    plt.savefig(output_file)  # pyright: ignore

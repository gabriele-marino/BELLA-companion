from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from bella_companion.backend.plots import ribbon_plot
from bella_companion.platyrrhine.plot.common import COLORS, RATE_LABELS, TYPE_LABELS
from bella_companion.platyrrhine.settings import (
    N_TIME_BINS,
    TIME_AXIS,
    TYPES,
)
from bella_companion.settings import settings


def plot_ribbons(output_file: Path):
    summaries_dir = settings.summaries_dir / "platyrrhine"
    summaries = pd.read_csv(summaries_dir / "summaries.csv")  # pyright: ignore

    _, axes = plt.subplots(  # pyright: ignore
        4, 3, figsize=(8, 10), sharey="all", sharex="col", layout="constrained"
    )

    for row, t in zip(axes, TYPES):
        for ax, rate in zip(row, ["birth", "death", "diversification"]):
            medians = summaries[
                [f"{rate}RateSPi{i}_{t}.median" for i in range(N_TIME_BINS)]
            ].values
            ribbon_plot(
                ax=ax,
                y=medians,
                x=TIME_AXIS,
                skyline=True,
                color=COLORS[rate][t],
                show_samples=False,
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (mya)")
        ax.invert_xaxis()
    for ax in axes[:, 0]:
        ax.set_ylabel("Rate")

    for col, rate in enumerate(["birth", "death", "diversification"]):
        handles = [
            Patch(facecolor=COLORS[rate][t], edgecolor="none", label=TYPE_LABELS[t])
            for t in TYPES
        ]
        axes[0, col].legend(
            title=f"Body mass ({RATE_LABELS[rate]})",  # pyright: ignore
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.75),
            title_fontsize=11,
            fontsize=10,
            handlelength=3,
            handleheight=1.5,
        )

    plt.savefig(output_file)  # pyright: ignore

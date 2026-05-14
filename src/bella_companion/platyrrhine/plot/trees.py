from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from phylogenie.draw import draw_colored_tree_continuous
from phylogenie.io import load_nexus

from bella_companion.platyrrhine.plot.common import RATE_LABELS
from bella_companion.settings import settings

_CMAPS: dict[str, LinearSegmentedColormap] = {
    "birth": plt.cm.Blues,  # pyright: ignore
    "death": plt.cm.Oranges,  # pyright: ignore
    "diversification": plt.cm.Greens,  # pyright: ignore
}


def plot_trees(output_file: Path):
    summaries_dir = settings.summaries_dir / "platyrrhine"

    _, axes = plt.subplots(1, 3, figsize=(12, 5), sharey="all", layout="constrained")  # pyright: ignore

    tree_file = summaries_dir / "mcc.nexus"
    tree = load_nexus(tree_file)["TREE_MCC_median"]
    tree.ladderize()
    for node in tree:
        node["diversificationRateSP"] = node["birthRateSP"] - node["deathRateSP"]

    for ax, rate in zip(axes, ["birth", "death", "diversification"]):
        cmap = LinearSegmentedColormap.from_list(
            "cmap",
            _CMAPS[rate](np.linspace(0.2, 1, 256)),  # pyright: ignore
        )
        _, hist_axes = draw_colored_tree_continuous(
            tree=tree,
            color_by=f"{rate}RateSP",
            ax=ax,
            backward_time=True,
            colormap=cmap,
            hist_axes_kwargs={
                "loc": "upper left",
                "bbox_to_anchor": (0.06, 0, 1, 1),
                "bbox_transform": ax.transAxes,
            },
        )
        hist_axes.set_xlabel(f"{RATE_LABELS[rate]}")  # pyright: ignore
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Time (mya)")
    plt.savefig(output_file)  # pyright: ignore

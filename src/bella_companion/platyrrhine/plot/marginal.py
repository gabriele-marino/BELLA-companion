from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from phylogenie.io import load_nexus
from phylogenie.tree_node import TreeNode

from bella_companion.backend.plots import ribbon_plot
from bella_companion.platyrrhine.plot.common import COLORS, RATE_LABELS
from bella_companion.platyrrhine.settings import N_TIME_BINS, TIME_AXIS
from bella_companion.settings import settings


def _get_marginal_rates(tree: TreeNode, rate: str) -> list[float]:
    ages = tree.ages

    for node in tree:
        node["diversificationRateSP"] = node["birthRateSP"] - node["deathRateSP"]

    def _node_is_in_time_bin(node: TreeNode, i: int) -> bool:
        bin_end = TIME_AXIS[i]
        bin_start = TIME_AXIS[i + 1]
        age = ages[node]
        origin = age + node.branch_length_or_raise()
        return bin_end > age >= bin_start or bin_end > origin >= bin_start

    return [
        np.mean(
            [node[f"{rate}RateSP"] for node in tree if _node_is_in_time_bin(node, i)],
            dtype=float,
        )
        for i in range(N_TIME_BINS)
    ]


def plot_marginal_rates(output_file: Path):
    summaries_dir = settings.summaries_dir / "platyrrhine"

    _, axes = plt.subplots(1, 3, figsize=(8, 3), sharey="all", layout="constrained")  # pyright: ignore

    for ax, rate in zip(axes, ["birth", "death", "diversification"]):
        ribbon_plot(
            ax=ax,
            y=[
                _get_marginal_rates(load_nexus(tree_file)["TREE_MCC_median"], rate)
                for tree_file in (summaries_dir / "mcc_trees").glob("*.nexus")
            ],
            x=TIME_AXIS,
            color=COLORS[rate].mean(axis=0),
            skyline=True,
            show_samples=False,
        )
        ax.invert_xaxis()
        ax.set_xlabel("Time (mya)")
    axes[0].set_ylabel("Rate")

    handles = [
        Patch(
            facecolor=COLORS[rate].mean(axis=0),
            edgecolor="none",
            label=RATE_LABELS[rate],
        )  # pyright: ignore
        for rate in ["birth", "death", "diversification"]
    ]
    axes[1].legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(handles),
        title_fontsize=11,
        fontsize=10,
        handlelength=3,
        handleheight=1.5,
    )

    plt.savefig(output_file)  # pyright: ignore

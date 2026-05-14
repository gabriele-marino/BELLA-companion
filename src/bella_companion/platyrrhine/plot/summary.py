from collections.abc import Iterable, Iterator
from itertools import product
from pathlib import Path

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from phylogenie.draw import draw_colored_tree_categorical
from phylogenie.io import load_nexus

from bella_companion.backend.activation_functions import Sigmoid
from bella_companion.backend.mlp import BayesMLP
from bella_companion.backend.plots import skyline_plot
from bella_companion.backend.xai import posterior_median_shap_importance
from bella_companion.platyrrhine.plot.common import COLORS, RATE_LABELS, TYPE_LABELS
from bella_companion.platyrrhine.settings import (
    N_TIME_BINS,
    RATE_UPPER,
    TIME_AXIS,
    TIME_BINS,
    TYPES,
)
from bella_companion.settings import settings
from bella_companion.typings import EnsembleModel, Model, PosteriorWeightsByMLP

_MEDIANS_LEGEND_LOC = {
    "birth": "lower left",
    "death": "lower center",
    "diversification": "upper center",
}


def plot_medians(ax: Axes, summaries: pd.DataFrame, rate: str, legend_loc: str):
    for t in TYPES:
        skyline_plot(
            summaries[
                [f"{rate}RateSPi{i}_{t}.median" for i in range(N_TIME_BINS)]
            ].median(),
            TIME_AXIS,
            ax=ax,
            step_kwargs={"color": COLORS[rate][t], "label": TYPE_LABELS[t]},
        )
    ax.legend(title="Body mass", loc=legend_loc, title_fontsize=9, fontsize=8)  # pyright: ignore


def plot_type_tree(ax: Axes):
    tree_file = settings.summaries_dir / "platyrrhine" / "mcc.nexus"
    tree = load_nexus(tree_file)["TREE_MCC_median"]
    tree.ladderize()
    for node in tree:
        node["diversificationRateSP"] = node["birthRateSP"] - node["deathRateSP"]
        node["color_by"] = int(
            node["type"] if "type" in node.metadata else node["type_median"]
        )

    cmap = ListedColormap(plt.cm.Purples(np.linspace(0.3, 1, 4)))  # pyright: ignore
    draw_colored_tree_categorical(
        ax=ax,
        tree=tree,
        color_by="color_by",
        backward_time=True,
        colormap=cmap,
        labels=TYPE_LABELS,
        legend_kwargs={"title": "Body mass", "loc": "upper left"},
        branch_kwargs={"linewidth": 2},
    )
    ax.set_xlabel("Time (mya)")  # pyright: ignore


def plot_shap(ax: Axes, models: Iterable[EnsembleModel], color: str):
    inputs = np.array(list(product(TIME_BINS, TYPES)))
    shap = posterior_median_shap_importance(models, inputs)
    shap /= shap.sum(axis=1, keepdims=True)
    for i, feature in enumerate(["Time", "Body mass"]):
        sns.violinplot(ax=ax, y=shap[:, i], x=[feature] * len(shap), cut=0, color=color)
    ax.set_ylim(0, 1)


class DiversificationMLP:
    def __init__(self, birth_mlps: EnsembleModel, death_mlps: EnsembleModel):
        self.birth_mlps = birth_mlps
        self.death_mlps = death_mlps

    def __iter__(self) -> Iterator[Model]:
        for birth_mlp, death_mlp in zip(self.birth_mlps, self.death_mlps):
            yield lambda x: birth_mlp(x) - death_mlp(x)


def plot_summary(output_file: Path):
    fig = plt.figure(figsize=(14, 8))  # pyright: ignore
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.5, 1, 1], wspace=0.3, hspace=0.1)

    summaries_dir = settings.summaries_dir / "platyrrhine"
    summaries = pd.read_csv(summaries_dir / "summaries.csv")  # pyright: ignore
    weights: list[PosteriorWeightsByMLP] = joblib.load(summaries_dir / "weights.pkl")

    models: dict[str, Iterable[EnsembleModel]] = {
        target: [
            BayesMLP(
                posterior_weights=w[f"{target}Rate"],
                output_activation=Sigmoid(RATE_UPPER),
            )
            for w in weights
        ]
        for target in ["birth", "death"]
    }
    models["diversification"] = [
        DiversificationMLP(mlp_birth, mlp_death)
        for mlp_birth, mlp_death in zip(models["birth"], models["death"])
    ]

    tree_ax = fig.add_subplot(gs[:, 0])
    plot_type_tree(tree_ax)
    tree_ax.text(  # pyright: ignore
        -0.1, 0.97, "a", transform=tree_ax.transAxes, fontsize=15, fontweight="bold"
    )

    for i, target in enumerate(["birth", "death", "diversification"]):
        ax = fig.add_subplot(gs[i, 1])
        if i == 0:
            ax.text(  # pyright: ignore
                -0.25,
                0.90,
                "b",
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
            )
        plot_medians(ax, summaries, target, _MEDIANS_LEGEND_LOC[target])
        ax.set_ylim((-0.1, 0.45))
        ax.set_ylabel(RATE_LABELS[target])  # pyright: ignore
        ax.invert_xaxis()  # pyright: ignore

        if i == 2:
            ax.set_xlabel("Time (mya)")  # pyright: ignore
        else:
            ax.tick_params(labelbottom=False)  # pyright: ignore

        ax = fig.add_subplot(gs[i, 2])
        if i == 0:
            ax.text(  # pyright: ignore
                -0.25,
                0.90,
                "c",
                transform=ax.transAxes,
                fontsize=15,
                fontweight="bold",
            )
        plot_shap(ax, models[target], f"C{i}")
        ax.set_ylabel("SHAP Importance")  # pyright: ignore

        if i == 2:
            ax.set_xlabel("Predictors")  # pyright: ignore
        else:
            ax.tick_params(labelbottom=False)  # pyright: ignore

    plt.savefig(output_file)  # pyright: ignore

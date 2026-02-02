import os
from collections.abc import Iterable, Sequence
from glob import glob
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from phylogenie import TreeNode, load_nexus
from phylogenie.draw import (
    draw_colored_tree_categorical,
    draw_colored_tree_continuous,
)

from bella_companion.backend import (
    MLPEnsemble,
    Model,
    Sigmoid,
    Weights,
    get_median_shap_feature_importance_distribution,
    normalize,
    ribbon_plot,
    skyline_plot,
)
from bella_companion.platyrrhine.settings import CHANGE_TIMES, TYPES

TYPE_LABELS = {0: "0 (Tiny)", 1: "1 (Small)", 2: "2 (Medium)", 3: "3 (Large)"}


def plot_platyrrhine_estimates():
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "platyrrhine"
    base_output_dir = (
        Path(os.environ["BELLA_FIGURES_DIR"]) / "platyrrhine" / "estimates"
    )

    log_summary = pd.read_csv(summaries_dir / "BELLA.csv")  # pyright: ignore
    for t in TYPES:
        for i in range(len(CHANGE_TIMES) + 1):
            log_summary[f"diversificationRateSPi{i}_{t}_median"] = (
                log_summary[f"birthRateSPi{i}_{t}_median"]
                - log_summary[f"deathRateSPi{i}_{t}_median"]
            )

    time_bins = list(reversed([0.0, *CHANGE_TIMES, 30.0]))

    gradient = np.linspace(0.4, 0.9, 4)
    colors: dict[str, np.typing.NDArray[np.floating]] = {
        "birth": plt.cm.Blues(gradient),  # pyright: ignore
        "death": plt.cm.Oranges(gradient),  # pyright: ignore
        "diversification": plt.cm.Greens(gradient),  # pyright: ignore
    }
    labels = {
        "birth": r"$\lambda$",
        "death": r"$\mu$",
        "diversification": r"$d$",
    }
    ribbons_ylims = {
        "birth": (0, 0.8),
        "death": (0, 0.6),
        "diversification": (-0.3, 0.5),
    }
    for rate in ["birth", "death", "diversification"]:
        output_dir = base_output_dir / rate
        os.makedirs(output_dir, exist_ok=True)

        # ----------------
        # Median estimates
        # ----------------

        for t, label in TYPE_LABELS.items():
            skyline_plot(
                [
                    log_summary[f"{rate}RateSPi{i}_{t}_median"].median()
                    for i in range(len(CHANGE_TIMES) + 1)
                ],
                time_bins,
                step_kwargs={"color": colors[rate][t], "label": label},
            )
        plt.gca().invert_xaxis()
        plt.legend(title="Body mass")  # pyright: ignore
        plt.xlabel("Time (mya)")  # pyright: ignore
        plt.ylabel(labels[rate])  # pyright: ignore
        if rate in {"birth", "death"}:
            plt.ylim(0, 0.4)  # pyright: ignore
        plt.savefig(output_dir / "all.svg")  # pyright: ignore
        plt.close()

        # ------------------
        # Marginal estimates
        # ------------------

        def _get_marginal_rates(tree: TreeNode) -> list[float]:
            ages = tree.ages
            for node in tree:
                node["diversificationRateSP"] = (
                    node["birthRateSP"] - node["deathRateSP"]
                )

            def _node_is_in_time_bin(node: TreeNode, i: int) -> bool:
                bin_end = time_bins[i]
                bin_start = time_bins[i + 1]
                age = ages[node]
                origin = age + node.branch_length_or_raise()
                return bin_end > age >= bin_start or bin_end > origin >= bin_start

            return [
                np.mean(
                    [
                        node[f"{rate}RateSP"]
                        for node in tree
                        if _node_is_in_time_bin(node, i)
                    ],
                    dtype=float,
                )
                for i in range(len(CHANGE_TIMES) + 1)
            ]

        ribbon_plot(
            [
                _get_marginal_rates(load_nexus(tree_file)["TREE_MCC_median"])
                for tree_file in glob(str(summaries_dir / "mcc_trees" / "*.nexus"))
            ],
            time_bins,
            color=colors[rate].mean(axis=0),
            skyline=True,
            samples_kwargs={"linewidth": 1},
        )
        plt.gca().invert_xaxis()
        plt.xlabel("Time (mya)")  # pyright: ignore
        plt.ylabel(labels[rate])  # pyright: ignore
        if rate in {"birth", "death"}:
            plt.ylim(0, 0.8)  # pyright: ignore
        plt.savefig(output_dir / "marginal.svg")  # pyright: ignore
        plt.close()

        # -------------------------
        # Ribbon plots by body mass
        # -------------------------

        for t in TYPES:
            ribbon_plot(
                log_summary[
                    [
                        f"{rate}RateSPi{i}_{t}_median"
                        for i in range(len(CHANGE_TIMES) + 1)
                    ]
                ].values,
                time_bins,
                color=colors[rate][t],
                skyline=True,
                samples_kwargs={"linewidth": 1},
            )
            plt.gca().invert_xaxis()
            plt.xlabel("Time (mya)")  # pyright: ignore
            plt.ylabel(labels[rate])  # pyright: ignore
            plt.ylim(*ribbons_ylims[rate])  # pyright: ignore
            plt.savefig(output_dir / f"body_mass={t}.svg")  # pyright: ignore
            plt.close()


def plot_platyrrhine_trees():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "platyrrhine" / "trees"
    os.makedirs(output_dir, exist_ok=True)

    tree_file = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "platyrrhine" / "mcc.nexus"
    tree = load_nexus(tree_file)["TREE_MCC_median"]
    tree.ladderize()
    for node in tree:
        node["diversificationRateSP"] = node["birthRateSP"] - node["deathRateSP"]

    plt.figure(figsize=(8, 11))  # pyright: ignore
    cmap = ListedColormap(plt.cm.Purples(np.linspace(0.3, 1, 4)))  # pyright: ignore
    ax = draw_colored_tree_categorical(
        tree=tree,
        color_by="type",
        backward_time=True,
        colormap=cmap,
        labels=TYPE_LABELS,
        legend_kwargs={"title": "Body mass", "loc": "upper left"},
    )
    ax.set_xlabel("Time (mya)")  # pyright: ignore
    plt.savefig(output_dir / "type.svg")  # pyright: ignore
    plt.close()

    cmaps: dict[str, LinearSegmentedColormap] = {
        "birthRateSP": plt.cm.Blues,  # pyright: ignore
        "deathRateSP": plt.cm.Oranges,  # pyright: ignore
        "diversificationRateSP": plt.cm.Greens,  # pyright: ignore
    }
    for color_by, cm in cmaps.items():
        cmap = LinearSegmentedColormap.from_list(
            "cmap",
            cm(np.linspace(0.2, 1, 256)),  # pyright: ignore
        )
        plt.figure(figsize=(8, 10))  #  pyright: ignore
        ax = plt.gca()
        ax, _ = draw_colored_tree_continuous(
            tree=tree,
            color_by=color_by,
            ax=ax,
            backward_time=True,
            colormap=cmap,
            hist_axes_kwargs={
                "loc": "upper left",
                "bbox_to_anchor": (0.06, 0, 1, 1),
                "bbox_transform": ax.transAxes,
            },
        )
        ax.set_xlabel("Time (mya)")  # pyright: ignore
        plt.savefig(output_dir / f"{color_by}.svg")  # pyright: ignore
        plt.close()


def plot_platyrrhine_shap():
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "platyrrhine"
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "platyrrhine" / "shap"
    os.makedirs(output_dir, exist_ok=True)

    weights: list[dict[str, list[Weights]]] = joblib.load(
        summaries_dir / "BELLA.weights.pkl"
    )
    mlps: dict[str, Sequence[Iterable[Model]]] = {
        rate: [
            MLPEnsemble(
                weights_list=sample_weights[f"{rate}Rate"],
                output_activation=Sigmoid(upper=5),
            )
            for sample_weights in weights
        ]
        for rate in ["birth", "death"]
    }
    mlps["diversification"] = [
        [
            lambda input: mlp_birth(input) - mlp_death(input)
            for mlp_birth, mlp_death in zip(mlps_birth, mlps_death)
        ]
        for mlps_birth, mlps_death in zip(mlps["birth"], mlps["death"])
    ]

    for rate, color in [("birth", "C0"), ("death", "C1"), ("diversification", "C2")]:
        inputs = list(product(normalize(CHANGE_TIMES), normalize(TYPES)))

        shap = get_median_shap_feature_importance_distribution(
            models=mlps[rate], inputs=inputs
        )
        shap /= shap.sum(axis=1, keepdims=True)
        for i, feature in enumerate(["Time", "Body mass"]):
            sns.violinplot(
                y=shap[:, i],
                x=[feature] * len(shap),
                cut=0,
                color=color,
            )
        plt.xlabel("Predictor")  # pyright: ignore
        plt.ylabel("Importance")  # pyright: ignore
        plt.savefig(output_dir / f"{rate}.svg")  # pyright: ignore
        plt.close()


def plot_platyrrhine():
    plot_platyrrhine_estimates()
    plot_platyrrhine_shap()
    plot_platyrrhine_trees()

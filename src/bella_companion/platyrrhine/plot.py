import os
from functools import partial
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from numpy.typing import NDArray
from phylogenie import draw_tree, get_node_depths, load_newick, load_nexus

from bella_companion.utils.explain import plot_shap_features_importance

DATA_DIR = Path(__file__).parent / "data"
# CHANGE_TIMES = (
#    pl.read_csv(DATA_DIR / "change_times.csv", has_header=False).to_series().to_list()
# )


def _plot_marginal_rates(output_dir: Path, trees_file: str | Path):
    trees = load_newick(trees_file)
    all_node_depths = [get_node_depths(tree) for tree in trees]
    max_time = max(max(node_depths.values()) for node_depths in all_node_depths)
    time_bins = np.array(list(reversed([0.0, *CHANGE_TIMES, max_time])))
    for tree in trees:
        for node in tree:
            node["diversificationRateSP"] = node["birthRateSP"] - node["deathRateSP"]

    for i, rate in enumerate(["birthRateSP", "deathRateSP", "diversificationRateSP"]):
        marginal_rates: list[list[float]] = [[] for _ in range(len(CHANGE_TIMES) + 1)]
        for tree, node_depths in zip(trees, all_node_depths):
            root_age = tree.height
            for node in tree:
                if node.parent is None:
                    marginal_rates[0].append(node[rate])
                    continue
                parent_age = root_age - node_depths[node.parent]  # pyright: ignore
                node_age = root_age - node_depths[node]
                t1 = np.where(time_bins >= parent_age)[0][-1]
                t2 = np.where(time_bins > node_age)[0][-1]
                for t in range(t1, t2 + 1):
                    marginal_rates[t].append(node[rate])

        median = [np.median(rates) for rates in marginal_rates]
        lower = [np.percentile(rates, 2.5) for rates in marginal_rates]
        upper = [np.percentile(rates, 97.5) for rates in marginal_rates]
        plt.step(  # pyright: ignore
            time_bins, [median[0], *median], color=f"C{i}", label=type
        )
        plt.fill_between(  # pyright: ignore
            time_bins,
            [lower[0], *lower],
            [upper[0], *upper],
            color=f"C{i}",
            alpha=0.2,
            step="pre",
        )

        plt.gca().invert_xaxis()
        plt.xlabel("Time (mya)")  # pyright: ignore
        plt.ylabel(  # pyright: ignore
            r"$\lambda$"
            if rate == "birthRateSP"
            else r"$\mu$"
            if rate == "deathRateSP"
            else r"$d$"
        )
        plt.savefig(output_dir / f"marginal-{rate}.svg")  # pyright: ignore
        plt.close()


def _plot_tree(output_dir: Path, tree_file: str | Path):
    sample_trees = load_nexus(tree_file)
    trees = [sample_trees[f"STATE_{i * 100_000}"] for i in range(11, 101)]

    for tree in trees:
        for node in tree:
            if node.parent is not None and not node.branch_length:
                node.parent.remove_child(node)

    avg_tree = trees[0].copy()
    for node in avg_tree:
        node.clear()
    for avg_node, nodes in zip(avg_tree, zip(*trees)):
        avg_node["birthRateSP"] = np.mean([n["birthRateSP"] for n in nodes])
        avg_node["deathRateSP"] = np.mean([n["deathRateSP"] for n in nodes])
        avg_node["diversificationRateSP"] = (
            avg_node["birthRateSP"] - avg_node["deathRateSP"]
        )
        avg_node["type"] = int(np.median([int(n["type"]) for n in nodes]))

    avg_tree.ladderize()
    cmaps: dict[str, LinearSegmentedColormap] = {
        "birthRateSP": plt.cm.Blues,  # pyright: ignore
        "deathRateSP": plt.cm.Oranges,  # pyright: ignore
        "diversificationRateSP": plt.cm.Greens,  # pyright: ignore
        "type": plt.cm.Purples,  # pyright: ignore
    }
    for color_by, cm in cmaps.items():
        ax = plt.gca()
        if color_by == "type":
            cmap = ListedColormap(cm(np.linspace(0.3, 1, 4)))  # pyright: ignore
            draw_tree(
                avg_tree,
                ax,
                color_by=color_by,
                legend_kwargs={"title": "Body mass", "loc": "upper left"},
                colormap=cmap,
            )
        else:
            cmap = LinearSegmentedColormap.from_list(
                "cmap",
                cm(np.linspace(0.2, 1, 256)),  # pyright: ignore
            )
            draw_tree(
                avg_tree,
                ax,
                color_by=color_by,
                hist_axes_kwargs={
                    "loc": "upper left",
                    "bbox_to_anchor": (0.06, 0, 1, 1),
                    "bbox_transform": ax.transAxes,
                },
                colormap=cmap,
            )
        height = avg_tree.height
        ticks = np.arange(height, 0, -10)
        ax.set_xticks(ticks, [str(round(height - t)) for t in ticks])  # pyright: ignore
        ax.set_xlabel("Time (mya)")  # pyright: ignore
        plt.savefig(output_dir / f"{color_by}-tree.svg")  # pyright: ignore
        plt.close()


def _plot_predictions(output_dir: Path, log_summary_file: str | Path):
    log_summary = pl.read_csv(log_summary_file)

    max_time = max(
        max(get_node_depths(tree).values())
        for tree in load_newick(DATA_DIR / "trees.nwk")
    )
    time_bins = list(reversed([0.0, *CHANGE_TIMES, max_time]))

    gradient = np.linspace(0.4, 0.9, 4)
    colors: dict[str, NDArray[np.floating]] = {
        "birthRateSP": plt.cm.Blues(gradient),  # pyright: ignore
        "deathRateSP": plt.cm.Oranges(gradient),  # pyright: ignore
        "diversificationRateSP": plt.cm.Greens(gradient),  # pyright: ignore
    }
    for rate in colors:
        for type in [0, 1, 2, 3]:
            if rate == "diversificationRateSP":
                estimates = log_summary.select(
                    [
                        pl.col(f"birthRateSPi{i}_{type}_median")
                        - pl.col(f"deathRateSPi{i}_{type}_median")
                        for i in range(len(CHANGE_TIMES) + 1)
                    ]
                ).to_numpy()
            else:
                estimates = log_summary.select(
                    [
                        pl.col(f"{rate}i{i}_{type}_median")
                        for i in range(len(CHANGE_TIMES) + 1)
                    ]
                ).to_numpy()
            median = np.median(estimates, axis=0)
            color = colors[rate][type]
            plt.step(  # pyright: ignore
                time_bins, [median[0], *median], color=color, label=type
            )

        plt.gca().invert_xaxis()
        plt.legend(title="Body mass")  # pyright: ignore
        plt.xlabel("Time (mya)")  # pyright: ignore
        plt.ylabel(  # pyright: ignore
            r"$\lambda$"
            if rate == "birthRateSP"
            else r"$\mu$"
            if rate == "deathRateSP"
            else r"$d$"
        )
        if rate in ["birthRateSP", "deathRateSP"]:
            plt.ylim(0, 0.4)  # pyright: ignore
        plt.savefig(output_dir / f"{rate}-predictions.svg")  # pyright: ignore
        plt.close()


def _plot_shap(output_dir: Path, weights_file: str | Path):
    weights = joblib.load(weights_file)
    for target, color in [("birthRate", "blue"), ("deathRate", "orange")]:
        plot_shap_features_importance(
            weights=[w[target] for w in weights],
            features={
                "Time": Feature(is_binary=False, color=color),
                "Body Mass": Feature(is_binary=False, color=color),
            },
            output_file=output_dir / f"shap-{target}.svg",
            hidden_activation=relu,
            output_activation=partial(sigmoid, upper=5),
        )


def plot_all():
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "platyrrhine"
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "platyrrhine"
    os.makedirs(output_dir, exist_ok=True)
    # _plot_predictions(output_dir, summaries_dir / "MLP.csv")
    # _plot_tree(output_dir, summaries_dir / "sample-trees.nexus")
    # _plot_marginal_rates(output_dir, summaries_dir / "trees.nwk")
    _plot_shap(output_dir, summaries_dir / "MLP.weights.pkl")

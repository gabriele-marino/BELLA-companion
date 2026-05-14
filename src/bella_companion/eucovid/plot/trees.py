import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from phylogenie.draw import draw_colored_dated_tree_categorical
from phylogenie.io import load_nexus

from bella_companion.eucovid.settings import COLORS
from bella_companion.settings import settings


def plot_trees(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    summaries_dir = settings.summaries_dir / "eucovid"

    for model in ["GLM", "BELLA"]:
        tree = load_nexus(summaries_dir / model / "mcc.nexus")["TREE_MCC_CA"]
        tree.ladderize()

        # If a node has multiple equally probable countries, randomly assign one for visualization
        rng = np.random.default_rng(42)
        for node in tree:
            if "+" in node["type"]:
                countries = node["type"].split("+")
                node["type"] = rng.choice(countries)

        node1 = (
            tree.get_descendant(name="CHN/WH-09/2020|China|2020-01-08"),
            date.fromisoformat("2020-01-08"),
        )
        node2 = (
            tree.get_descendant(name="IMS-10216-CVDP-0161|Germany|2020-03-08"),
            date.fromisoformat("2020-03-08"),
        )

        plt.figure(figsize=(5, 9), layout="constrained")  # pyright: ignore

        ax = draw_colored_dated_tree_categorical(
            tree=tree,
            calibration_nodes=(node1, node2),
            color_by="type",
            colormap=COLORS,
            legend_kwargs={"loc": "upper left"},
            branch_kwargs={"linewidth": 3},
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.savefig(output_dir / f"{model}.pdf")  # pyright: ignore

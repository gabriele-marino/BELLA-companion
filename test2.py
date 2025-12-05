from datetime import datetime
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from phylogenie import (
    Tree,
    draw_tree,
    get_node_depth_levels,
    get_node_depths,
    load_nexus,
)
from phylogenie.draw import Color, _draw_colored_tree

plt.rcParams["font.family"] = "Arial"


def draw_tree_from_tip_dates(
    tree: Tree,
    min_tip_date: datetime,
    max_tip_date: datetime,
    colors: dict[Tree, Color],
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    xs = (
        get_node_depth_levels(tree)
        if any(node.branch_length is None for node in tree.iter_descendants())
        else get_node_depths(tree)
    )
    tip_depths = [d for n, d in xs.items() if n.is_leaf()]
    min_depth = min(tip_depths)
    max_depth = max(tip_depths)

    min_tip_date_num = mdates.date2num(min_tip_date)  # pyright: ignore
    max_tip_date_num = mdates.date2num(max_tip_date)  # pyright: ignore

    # Step 3: linear mapping depth -> date
    def depth_to_date(depth: float) -> datetime:
        return min_tip_date_num + (depth - min_depth) * (
            max_tip_date_num - min_tip_date_num
        ) / (max_depth - min_depth)

    xs_dates = {node: depth_to_date(depth) for node, depth in xs.items()}

    _draw_colored_tree_with_xs(tree, ax, colors, xs_dates)

    # Step 5: format x-axis as dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)

    return ax


def _draw_colored_tree_with_xs(
    tree: Tree,
    ax: Axes,
    colors: Color | dict[Tree, Color],
    xs_dates: dict[Tree, datetime],
) -> Axes:
    if not isinstance(colors, dict):
        colors = {node: colors for node in tree}

    ys: dict[Tree, float] = {node: i for i, node in enumerate(tree.get_leaves())}
    for node in tree.postorder_traversal():
        if node.is_internal():
            ys[node] = sum(ys[child] for child in node.children) / len(node.children)

    for node in tree:
        x1, y1 = xs_dates[node], ys[node]
        if node.parent is None:
            continue
        #    ax.hlines(y=y1, xmin=0, xmax=x1, color=colors[node])
        #    continue
        x0, y0 = xs_dates[node.parent], ys[node.parent]
        ax.vlines(x=x0, ymin=y0, ymax=y1, color=colors[node])
        ax.hlines(y=y1, xmin=x0, xmax=x1, color=colors[node])

    ax.set_yticks([])
    return ax


plt.figure(figsize=(6, 8))
colors = {
    "China": "#F0E442",
    "France": "#009E73",
    "Germany": "#D55E00",
    "Italy": "#E69F00",
    "OtherEU": "#56B4E9",
}
tree = load_nexus(
    "/Users/gmarino/bella_companion/outputs/runs/eucovid/flights_over_population/GLM/TypedNodeTrees.annotated.trees"
)["TREE_MCC_CA"]
tree.ladderize()

ax = draw_tree_from_tip_dates(
    tree,
    min_tip_date=datetime.strptime("2019-12-26", "%Y-%m-%d"),
    max_tip_date=datetime.strptime("2020-03-08", "%Y-%m-%d"),
    colors={node: colors[node["type"]] for node in tree},
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("tree.svg")
plt.show()

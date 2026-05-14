from collections.abc import Iterable

import pandas as pd
from matplotlib.axes import Axes

from bella_companion.simulations.plot.common import plot_medians_interquantile
from bella_companion.simulations.scenarios import FBD_TIME_AXIS
from bella_companion.targets import SkylineTarget

STATE_COLORS = {
    "00": "#0072B2",
    "01": "#009E73",
    "10": "#CC79A7",
    "11": "#E69F00",
}


def plot_estimate_interquantiles(
    ax: Axes,
    summaries: pd.DataFrame,
    targets: Iterable[SkylineTarget],
):
    for target in targets:
        assert target.state is not None
        s1, _ = target.state
        plot_medians_interquantile(
            ax=ax,
            summaries=summaries,
            time_axis=FBD_TIME_AXIS,
            target=target,
            ribbon_color=STATE_COLORS[target.state],
            true_value_color="black" if int(s1) else "gray",
            times_are_ages=True,
        )
    ax.invert_xaxis()

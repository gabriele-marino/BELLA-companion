from collections.abc import Mapping

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from bella_companion.backend.plots import Color, ribbon_plot, skyline_plot
from bella_companion.metrics import PerRunMetric, PerStudyMetric
from bella_companion.settings import settings
from bella_companion.targets import SkylineTarget
from bella_companion.typings import ArrayLike, ModelID


def plot_per_run_metric_through_time(
    ax: Axes,
    metric: PerRunMetric,
    target: SkylineTarget,
    models_summaries: Mapping[ModelID, pd.DataFrame],
    times_are_ages: bool,
):
    data = pd.concat(
        [
            pd.DataFrame({metric.name: values})
            .assign(model=model)
            .assign(**{"Time bin": target.n_time_bins - i if times_are_ages else i + 1})
            for model, summaries in models_summaries.items()
            for i, values in enumerate(metric(summaries, target).T)
        ]
    )

    sns.violinplot(
        x="Time bin",
        y=metric.name,
        hue="model",
        data=data,
        inner=None,
        cut=0,
        density_norm="width",
        palette=settings.model_colors,
        legend=False,
        ax=ax,
    )


def plot_per_study_metric_through_time(
    ax: Axes,
    metric: PerStudyMetric,
    target: SkylineTarget,
    models_summaries: Mapping[ModelID, pd.DataFrame],
    times_are_ages: bool,
):
    for model, summaries in models_summaries.items():
        values = metric(summaries, target)
        if times_are_ages:
            values = list(reversed(values))
        ax.plot(  # pyright: ignore
            np.arange(1, target.n_time_bins + 1),
            values,
            marker="o",
            color=settings.model_colors[model],
        )
    ax.set_ylabel(metric.name)  # pyright: ignore
    ax.set_ylim((0, 1.05))  # pyright: ignore


def plot_medians_ribbon(
    ax: Axes,
    time_axis: ArrayLike,
    target: SkylineTarget,
    summaries: pd.DataFrame,
    color: Color,
    times_are_ages: bool = False,
):
    medians = summaries[[f"{key}.median" for key in target.keys]].values
    if times_are_ages:
        medians = [list(reversed(median)) for median in medians]
    rates = list(reversed(target.values)) if times_are_ages else target.values
    ribbon_plot(
        y=medians, skyline=True, x=time_axis, ax=ax, color=color, show_samples=False
    )
    skyline_plot(
        rates,
        x=time_axis,
        ax=ax,
        step_kwargs={"color": "k", "linestyle": "--"},
    )


def plot_medians_interquantile(
    ax: Axes,
    summaries: pd.DataFrame,
    time_axis: ArrayLike,
    target: SkylineTarget,
    ribbon_color: Color,
    true_value_color: Color,
    times_are_ages: bool,
):
    medians = summaries[[f"{key}.median" for key in target.keys]].values
    if times_are_ages:
        medians = [list(reversed(median)) for median in medians]
    ribbon_plot(
        x=time_axis,
        y=medians,
        skyline=True,
        color=ribbon_color,
        show_samples=False,
        percentiles=(50,),
        median_kwargs={"linewidth": 2},
        ax=ax,
    )
    rate_values = list(reversed(target.skyline)) if times_are_ages else target.skyline
    skyline_plot(
        rate_values,
        x=time_axis,
        step_kwargs={"color": true_value_color, "linestyle": "--"},
        ax=ax,
    )

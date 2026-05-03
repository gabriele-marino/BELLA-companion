import string
from typing import TypeVar

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.metrics import (
    CoefficientOfVariation,
    Coverage,
    MeanESSPerHour,
    Metric,
    NormalizedMAE,
)
from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS

T = TypeVar("T")


def _plot_metric(metric: Metric, sharex: bool = False, log_xscale: bool = False):
    _, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")  # pyright: ignore

    for ax, label in zip(axes.flat, string.ascii_lowercase):
        ax.text(
            -0.07, 1.02, label, transform=ax.transAxes, fontsize=15, fontweight="bold"
        )

    models = ["PA", "GLM", *settings.bella_model_configs]
    for ax, (scenario_id, scenario) in zip(axes.flat, SCENARIOS.items()):
        models_summaries = {
            model: pd.read_csv(settings.summaries_dir / scenario_id / f"{model}.csv")  # pyright: ignore
            for model in models
        }

        ys = list(range(len(models)))
        for y, summaries in zip(ys, reversed(models_summaries.values())):
            metric.plot(ax, y, summaries, scenario.targets)

        ax.set_yticks(ys)
        ax.set_yticklabels(list(reversed(models)))
        if log_xscale:
            ax.set_xscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

    for ax in axes[1, :]:
        ax.set_xlabel(metric.name)
    if sharex:
        for ax in axes[0, :]:
            ax.tick_params(labelbottom=False)
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)

    plt.savefig(settings.figures_dir / f"{metric.id}.svg")  # pyright: ignore


def plot_metrics():
    _plot_metric(metric=NormalizedMAE())
    _plot_metric(metric=Coverage(), sharex=True)
    _plot_metric(metric=CoefficientOfVariation())
    _plot_metric(metric=MeanESSPerHour(), log_xscale=True)

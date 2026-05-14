import os
import string
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bella_companion.metrics import (
    MAPE,
    CoefficientOfVariation,
    Coverage,
    MeanESSPerHour,
    Metric,
)
from bella_companion.settings import settings
from bella_companion.simulations.scenarios import SCENARIOS


def _plot_metric(
    metric: Metric, output_dir: Path, sharex: bool = False, log_xscale: bool = False
):
    _, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")  # pyright: ignore

    for ax, label in zip(axes.flat, string.ascii_lowercase):
        ax.text(
            -0.07, 1.02, label, transform=ax.transAxes, fontsize=15, fontweight="bold"
        )

    models = list(reversed(["PA", "GLM", *settings.bella_model_configs]))
    for ax, (scenario_id, scenario) in zip(axes.flat, SCENARIOS.items()):
        models_summaries = {
            model: pd.read_csv(settings.summaries_dir / scenario_id / f"{model}.csv")  # pyright: ignore
            for model in models
        }
        metric.plot(ax, models_summaries, scenario.targets)

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

    plt.savefig(output_dir / f"{metric.id}.pdf")  # pyright: ignore


def plot_metrics(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    _plot_metric(MAPE(), output_dir)
    _plot_metric(Coverage(), output_dir, sharex=True)
    _plot_metric(CoefficientOfVariation(), output_dir)
    _plot_metric(MeanESSPerHour(), output_dir, log_xscale=True)

import os
from functools import partial
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bella_companion.simulations.scenarios.fbd_2traits import (
    FBD_RATE_UPPER,
    N_TIME_BINS,
    RATES,
    SCENARIO,
    STATES,
)
from bella_companion.utils.explain.pdp import plot_partial_dependencies


def _set_time_bin_xticks(n: int, reverse: bool = False):
    xticks_labels = range(n)
    if reverse:
        xticks_labels = reversed(xticks_labels)
    plt.xticks(ticks=range(n), labels=list(map(str, xticks_labels)))  # pyright: ignore
    plt.xlabel("Time bin")  # pyright: ignore


def step(
    x: list[float],
    reverse_xticks: bool = False,
    label: str | None = None,
    color: str | None = None,
    linestyle: str | None = None,
):
    x = [x[0], *x]
    n = len(x)
    plt.step(  # pyright: ignore
        list(range(n)), x, label=label, color=color, linestyle=linestyle
    )
    _set_time_bin_xticks(n, reverse_xticks)


def _plot_predictions(log_summary: pd.DataFrame):
    for rate, state_rates in RATES.items():
        label = r"\lambda" if rate == "birth" else r"\mu"
        for state in STATES:
            estimates = [
                float(np.median(log_summary[f"{rate}RateSPi{i}_{state}_median"]))
                for i in range(N_TIME_BINS)
            ]
            step(
                estimates,
                label=rf"${label}_{{{state[0]},{state[1]}}}$",
                reverse_xticks=True,
            )
        step(
            state_rates["00"],
            color="k",
            linestyle="dashed",
            reverse_xticks=True,
        )
        step(
            state_rates["10"],
            color="gray",
            linestyle="dashed",
            reverse_xticks=True,
        )
        plt.legend()  # pyright: ignore
        plt.ylabel(rf"${label}$")  # pyright: ignore
        plt.savefig(f"{rate}-predictions.svg")  # pyright: ignore
        plt.close()


def plot_fbd_2traits_results():
    log_summary = pd.read_csv(
        "/Users/gmarino/bella_companion/outputs/beast/summaries/fbd-2traits/GLM.csv"
    )

    _plot_predictions(log_summary)


plot_fbd_2traits_results()

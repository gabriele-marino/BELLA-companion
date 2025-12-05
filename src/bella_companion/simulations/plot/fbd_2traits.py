import os
from functools import partial
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bella_companion.simulations.plot.utils import step
from bella_companion.simulations.scenarios.fbd_2traits import (
    FBD_RATE_UPPER,
    N_TIME_BINS,
    RATES,
    SCENARIO,
    STATES,
)


def _plot_predictions(log_summary: pd.DataFrame, output_dir: Path):
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
        plt.savefig(output_dir / rate / "predictions.svg")  # pyright: ignore
        plt.close()


def plot_fbd_2traits_results():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "fbd-2traits"

    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "fbd-2traits"
    model = "MLP-32_16"
    log_summary = pl.read_csv(summaries_dir / f"{model}.csv")
    weights = joblib.load(summaries_dir / f"{model}.weights.pkl")

    for rate in RATES:
        os.makedirs(output_dir / rate, exist_ok=True)
        plot_partial_dependencies(
            weights=[w[f"{rate}Rate"] for w in weights],
            features=SCENARIO.features[f"{rate}Rate"],
            output_dir=output_dir / rate,
            hidden_activation=relu,
            output_activation=partial(sigmoid, upper=FBD_RATE_UPPER),
        )
        plot_shap_features_importance(
            weights=[w[f"{rate}Rate"] for w in weights],
            features=SCENARIO.features[f"{rate}Rate"],
            output_file=output_dir / rate / "shap_values.svg",
            hidden_activation=relu,
            output_activation=partial(sigmoid, upper=FBD_RATE_UPPER),
        )

    _plot_predictions(log_summary, output_dir)

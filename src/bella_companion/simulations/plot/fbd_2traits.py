import os
from itertools import chain, product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bella_companion.backend import (
    MEDIAN_POSTFIX,
    MLPEnsemble,
    Sigmoid,
    Weights,
    get_median_partial_dependence_plot_distribution,
    get_median_shap_feature_importance_distribution,
    skyline_plot,
)
from bella_companion.backend.plots import ribbon_plot
from bella_companion.simulations.scenarios.fbd_2traits import (
    FBD_RATE_UPPER,
    N_TIME_BINS,
    RATES,
    STATES,
)


def plot_fbd_2traits():
    base_output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "fbd-2traits"
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "fbd-2traits"
    model = "BELLA-16_8"
    summaries = pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
    weights: list[dict[str, list[Weights]]] = joblib.load(
        summaries_dir / f"{model}.weights.pkl"
    )

    for rate, state_rates in RATES.items():
        output_dir = base_output_dir / rate
        os.makedirs(output_dir, exist_ok=True)

        label = r"\lambda" if rate == "birth" else r"\mu"
        for state, color in zip(STATES, ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]):
            estimates = [
                summaries[f"{rate}RateSPi{i}_{state}{MEDIAN_POSTFIX}"].median()
                for i in range(N_TIME_BINS)
            ]
            skyline_plot(
                list(reversed(estimates)),
                step_kwargs={
                    "label": rf"${label}_{{{state[0]},{state[1]}}}$",
                    "color": color,
                },
            )
        skyline_plot(
            list(reversed(state_rates["00"])),
            step_kwargs={"color": "k", "linestyle": "dashed"},
        )
        skyline_plot(
            list(reversed(state_rates["10"])),
            step_kwargs={"color": "gray", "linestyle": "dashed"},
        )
        plt.gca().invert_xaxis()
        plt.legend()  # pyright: ignore
        plt.xlabel("Time")  # pyright: ignore
        plt.ylabel(rf"${label}$")  # pyright: ignore
        plt.savefig(output_dir / "predictions.svg")  # pyright: ignore
        plt.close()

        mlp_ensembles = [
            MLPEnsemble(
                weights_list=sample_weights[f"{rate}Rate"],
                output_activation=Sigmoid(upper=FBD_RATE_UPPER),
            )
            for sample_weights in weights
        ]
        inputs = list(
            product(
                np.linspace(0, 1, 10),
                [0, 1],
                [0, 1],
                np.linspace(0, 1, 10),
            )
        )

        for feature_idx, feature, color in [(3, "Random", "gray"), (0, "Time", "red")]:
            grid = np.linspace(0, 1, 10).tolist()
            pdps = get_median_partial_dependence_plot_distribution(
                models=mlp_ensembles,
                inputs=inputs,
                feature_idx=feature_idx,
                grid=grid,
            )
            ribbon_plot(
                x=grid,
                y=pdps,
                color=color,
                label=feature,
                samples_kwargs={"linewidth": 1},
            )
        plt.xlabel("Predictor value")  # pyright: ignore
        plt.ylabel(rf"Marginal ${label}$")  # pyright: ignore
        plt.legend()  # pyright: ignore
        plt.savefig(output_dir / "PDP-continuous.svg")  # pyright: ignore
        plt.close()

        binary_features = [(1, "Trait 1", "red"), (2, "Trait 2", "gray")]
        data: list[float] = []
        x: list[float] = []
        labels: list[str] = []
        for feature_idx, feature, color in binary_features:
            grid = [0, 1]
            pdps = get_median_partial_dependence_plot_distribution(
                models=mlp_ensembles,
                inputs=inputs,
                feature_idx=feature_idx,
                grid=grid,
            )
            data.extend(list(chain(*pdps)))
            x.extend(grid * len(pdps))
            labels.extend([feature] * (2 * len(pdps)))
        ax = sns.boxplot(x=labels, y=data, hue=x)
        ax.get_legend().remove()  # pyright: ignore
        for i, (feature_idx, feature, color) in enumerate(binary_features):
            ax.patches[i].set_facecolor(color)
            ax.patches[i + len(binary_features)].set_facecolor(color)
        plt.xlabel("Predictor")  # pyright: ignore
        plt.ylabel(rf"Marginal ${label}$")  # pyright: ignore
        plt.savefig(output_dir / "PDPs-categorical.svg")  # pyright: ignore
        plt.close()

        shap = get_median_shap_feature_importance_distribution(
            models=mlp_ensembles, inputs=inputs
        )
        shap /= shap.sum(axis=1, keepdims=True)
        for feature_idx, feature, color in [
            (0, "Time", "red"),
            (3, "Random", "gray"),
            (1, "Trait 1", "red"),
            (2, "Trait 2", "gray"),
        ]:
            sns.violinplot(
                y=shap[:, feature_idx],
                x=[feature] * len(shap),
                cut=0,
                color=color,
            )
        plt.xlabel("Predictor")  # pyright: ignore
        plt.ylabel("Importance")  # pyright: ignore
        plt.savefig(output_dir / "shap.svg")  # pyright: ignore
        plt.close()

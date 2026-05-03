import os
import string
from collections.abc import Iterable
from itertools import chain, product

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from bella_companion.backend.activation_functions import Sigmoid
from bella_companion.backend.mlp import BayesMLP
from bella_companion.backend.plots import ribbon_plot
from bella_companion.backend.xai import (
    posterior_median_pdp,
    posterior_median_shap_importance,
)
from bella_companion.settings import settings
from bella_companion.simulations.plot.common import plot_medians_interquantile
from bella_companion.simulations.scenarios import (
    FBD_2TRAITS,
    FBD_RATE_UPPER,
    FBD_TIME_AXIS,
    N_TIME_BINS,
)
from bella_companion.targets import SkylineTarget
from bella_companion.typings import PosteriorWeightsByMLP

STATE_COLORS = {
    "00": "#0072B2",
    "01": "#009E73",
    "10": "#CC79A7",
    "11": "#E69F00",
}


def _plot_medians_interquantile(
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


def plot_summary():
    summaries_dir = settings.summaries_dir / "fbd-2traits"
    reference_model = settings.bella_reference_models["fbd-2traits"]
    summaries = pd.read_csv(summaries_dir / f"{reference_model}.csv")  # pyright: ignore
    weights: list[PosteriorWeightsByMLP] = joblib.load(
        summaries_dir / f"{reference_model}.weights.pkl"
    )

    output_dir = settings.figures_dir / "fbd-2traits"
    os.makedirs(output_dir, exist_ok=True)

    _, axes = plt.subplots(2, 4, figsize=(13, 6), sharex="col", layout="constrained")  # pyright: ignore

    for col, letter in zip([0, 2, 3], string.ascii_lowercase):
        ax = axes[0, col]
        ax.text(
            -0.225, 0.95, letter, transform=ax.transAxes, fontsize=15, fontweight="bold"
        )

    inputs = np.array(
        list(product(np.linspace(0, 1, N_TIME_BINS), [0, 1], [0, 1], [0, 1]))
    )
    for i, (rate_name, rates) in enumerate(
        [("birth", FBD_2TRAITS.birth_rates), ("death", FBD_2TRAITS.death_rates)]
    ):
        bayes_mlps = [
            BayesMLP(
                posterior_weights=run_weights[f"{rate_name}Rate"],
                hidden_activation=settings.bella_model_configs[
                    reference_model
                ].hidden_activation,
                output_activation=Sigmoid(upper=FBD_RATE_UPPER),
            )
            for run_weights in weights
        ]

        death_predictor_color = "red" if rate_name == "death" else "gray"
        time_predictor_color = "gray" if rate_name == "death" else "red"

        grid = np.linspace(0, 1, N_TIME_BINS + 1).tolist()
        pdps = posterior_median_pdp(
            bayes_mlps=bayes_mlps,
            inputs=inputs,
            feature_idx=0,
            grid=grid,
        )
        ribbon_plot(
            ax=axes[i, 0],
            x=FBD_TIME_AXIS,
            y=[list(reversed(pdp)) for pdp in pdps],
            color=time_predictor_color,
            samples_kwargs={"linewidth": 1},
        )

        binary_features = [
            (1, "Time series", death_predictor_color),
            (2, "Trait 1", "red"),
            (3, "Trait 2", "gray"),
        ]
        data: list[float] = []
        x: list[float] = []
        labels: list[str] = []
        for feature_idx, feature, color in binary_features:
            grid = [0, 1]
            pdps = posterior_median_pdp(
                bayes_mlps=bayes_mlps,
                inputs=inputs,
                feature_idx=feature_idx,
                grid=np.array(grid),
            )
            data.extend(list(chain(*pdps)))
            x.extend(grid * len(pdps))
            labels.extend([feature] * (2 * len(pdps)))
        sns.boxplot(x=labels, y=data, hue=x, ax=axes[i, 1])
        axes[i, 1].get_legend().remove()  # pyright: ignore
        for j, (feature_idx, feature, color) in enumerate(binary_features):
            axes[i, 1].patches[j].set_facecolor(color)
            axes[i, 1].patches[j + len(binary_features)].set_facecolor(color)

        shap = posterior_median_shap_importance(bayes_mlps=bayes_mlps, inputs=inputs)
        shap /= shap.sum(axis=1, keepdims=True)
        for feature_idx, feature, color in [
            (0, "Time", time_predictor_color),
            (1, "Time series", death_predictor_color),
            (2, "Trait 1", "red"),
            (3, "Trait 2", "gray"),
        ]:
            sns.violinplot(
                y=shap[:, feature_idx],
                x=[feature] * len(shap),
                cut=0,
                color=color,
                ax=axes[i, 2],
            )
        axes[i, 2].set_ylabel("SHAP Importance")

        _plot_medians_interquantile(ax=axes[i, 3], summaries=summaries, targets=rates)

    for col_idx in range(2):
        axes[0, col_idx].set_ylim((0.1, 0.7))
        axes[1, col_idx].set_ylim((0, 1.45))

    axes[0, 0].set_ylabel(r"Marginal $\lambda$")
    axes[1, 0].set_ylabel(r"Marginal $\mu$")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].invert_xaxis()

    axes[1, 1].set_xlabel("Predictors")
    axes[1, 2].set_xlabel("Predictors")

    axes[1, 3].set_xlabel("Time")
    axes[1, 3].invert_xaxis()

    plt.savefig(output_dir / "summary.svg")  # pyright: ignore

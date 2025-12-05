import os
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm


def _get_median_partial_dependence_values(
    # weights: list[Weights],  # shape: (n_weight_samples, ...)
    # inputs: ArrayLike,
    # features_grid: list[list[float]],
    # hidden_activation: ActivationFunction,
    # output_activation: ActivationFunction,
) -> list[list[float]]:  # shape: (n_features, n_grid_points)
    pdvalues = [
        get_partial_dependence_values(
            weights=w,
            inputs=inputs,
            features_grid=features_grid,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        for w in weights
    ]
    return [
        np.median([pd[feature_idx] for pd in pdvalues], axis=0).tolist()
        for feature_idx in range(len(features_grid))
    ]


def plot_partial_dependencies(
    # weights: list[Weights] | list[list[Weights]],
    # features: dict[str, Feature],
    # hidden_activation: ActivationFunction,
    # output_activation: ActivationFunction,
    # inputs: ArrayLike | None = None,
    # y_log_scale: bool = False,
    # output_dir: str | Path | None = None,
):
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    continuous_grid: list[float] = np.linspace(0, 1, 10).tolist()
    features_grid: list[list[float]] = [
        [0, 1] if feature.is_binary else continuous_grid
        for feature in features.values()
    ]
    if inputs is None:
        inputs = list(product(*features_grid))

    if tg.is_weights_list(weights):
        func = get_partial_dependence_values
    elif tg.is_nested_weights_list(weights):
        func = _get_median_partial_dependence_values
    else:
        raise ValueError(
            "weights must be a list of Weights or a list of list of Weights"
        )

    jobs = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(
            partial(
                func,
                inputs=inputs,
                features_grid=features_grid,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
            )
        )(w)
        for w in weights
    )
    pdvalues = [
        job for job in tqdm(jobs, total=len(weights), desc="Evaluating PDPs")
    ]  # shape: (n_runs, n_features, n_grid_points)
    pdvalues = [
        np.array(mcmc_pds).T for mcmc_pds in zip(*pdvalues)
    ]  # shape: (n_features, n_grid_points, n_runs)

    binary_features = [f for f in features.values() if f.is_binary]
    if binary_features:
        data: list[float] = []
        grid: list[int] = []
        labels: list[str] = []
        for (label, feature), feature_pdvalues in zip(features.items(), pdvalues):
            if feature.is_binary:
                for i in [0, 1]:
                    data.extend(feature_pdvalues[i])
                    grid.extend([i] * len(feature_pdvalues[i]))
                    labels.extend([label] * len(feature_pdvalues[i]))

        ax = sns.boxplot(x=labels, y=data, hue=grid)
        ax.get_legend().remove()  # pyright: ignore

        for i, feature in enumerate(binary_features):
            ax.patches[i].set_facecolor(feature.color)
            ax.patches[i + len(binary_features)].set_facecolor(feature.color)

        plt.xlabel("Predictor")  # pyright: ignore
        plt.ylabel("MLP Output")  # pyright: ignore
        if y_log_scale:
            plt.yscale("log")  # pyright: ignore
        plt.savefig(output_dir / "PDPs-categorical.svg")  # pyright: ignore
        plt.close()

    if len(features) > len(binary_features):
        for (label, feature), feature_pdvalues in zip(features.items(), pdvalues):
            if not feature.is_binary:
                median = np.median(feature_pdvalues, axis=1)
                lower = np.percentile(feature_pdvalues, 2.5, axis=1)
                high = np.percentile(feature_pdvalues, 100 - 2.5, axis=1)
                plt.fill_between(  # pyright: ignore
                    continuous_grid, lower, high, alpha=0.25, color=feature.color
                )
                for mcmc_pds in feature_pdvalues.T:
                    plt.plot(  # pyright: ignore
                        continuous_grid,
                        mcmc_pds,
                        color=feature.color,
                        alpha=0.2,
                        linewidth=1,
                    )
                plt.plot(  # pyright: ignore
                    continuous_grid, median, color=feature.color, label=label
                )
        plt.xlabel("Predictor value")  # pyright: ignore
        plt.ylabel("MLP Output")  # pyright: ignore
        if y_log_scale:
            plt.yscale("log")  # pyright: ignore
        plt.legend()  # pyright: ignore
        plt.savefig(output_dir / "PDPs-continuous.svg")  # pyright: ignore
        plt.close()

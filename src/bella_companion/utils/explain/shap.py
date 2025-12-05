from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm


def _get_median_shap_features_importance(
    # weights: list[Weights],
    # inputs: ArrayLike,
    # hidden_activation: ActivationFunction,
    # output_activation: ActivationFunction,
) -> list[float]:  # length: n_features
    features_importance = np.array(
        [
            get_shap_features_importance(
                weights=w,
                inputs=inputs,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
            )
            for w in weights
        ]
    )  # shape: (n_samples, n_features)
    features_importance /= features_importance.sum(axis=1, keepdims=True)
    return np.median(features_importance, axis=0).tolist()


def _plot_shap_violins(
    # features: dict[str, Feature],
    # features_importance: Array,  # shape: (n_samples, n_features)
    # output_file: str | Path,
):
    def _plot_violins(group_check: Callable[[Feature], bool]):
        for i, (feature_name, feature) in enumerate(features.items()):
            if group_check(feature):
                sns.violinplot(
                    y=features_importance[:, i],
                    x=[feature_name] * len(features_importance),
                    cut=0,
                    color=feature.color,
                )

    _plot_violins(lambda f: not f.is_binary)
    _plot_violins(lambda f: f.is_binary)
    plt.xlabel("Predictor")  # pyright: ignore
    plt.ylabel("Importance")  # pyright: ignore
    plt.savefig(output_file)  # pyright: ignore
    plt.close()


def plot_shap_features_importance(
    # weights: list[Weights] | list[list[Weights]],
    # features: dict[str, Feature],
    # output_file: str | Path,
    # hidden_activation: ActivationFunction,
    # output_activation: ActivationFunction,
    # inputs: ArrayLike | None = None,
):
    if inputs is None:
        continuous_grid: list[float] = np.linspace(0, 1, 10).tolist()
        features_grid: list[list[float]] = [
            [0, 1] if feature.is_binary else continuous_grid
            for feature in features.values()
        ]
        inputs = list(product(*features_grid))

    if tg.is_weights_list(weights):
        func = get_shap_features_importance
    elif tg.is_nested_weights_list(weights):
        func = _get_median_shap_features_importance
    else:
        raise ValueError(
            "weights must be a list of Weights or a list of list of Weights"
        )

    jobs = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(
            partial(
                func,
                inputs=inputs,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
            )
        )(w)
        for w in weights
    )
    features_importance = np.array(
        [job for job in tqdm(jobs, total=len(weights), desc="Evaluating SHAPs")]
    )  # shape: (n_runs, n_features)
    _plot_shap_violins(features, features_importance, output_file)

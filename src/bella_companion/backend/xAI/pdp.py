from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from bella_companion.backend.mlp import MLP, MLPEnsemble


def get_partial_dependence_plot(
    mlp: MLP,
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Sequence[float],  # [...n_grid_points]
) -> list[float]:  # [...n_grid_points]
    """
    Compute partial dependence values for a single feature over a specified grid.

    Parameters
    ----------
    mlp : MLP
        An instance of the MLP class representing the multi-layer perceptron model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Sequence[float]
        A list of grid points for the specified feature.

    Returns
    -------
    list[float]
        A list containing the partial dependence values for the specified feature
        over the given grid points.
    """
    pdvalues: list[float] = []
    for grid_point in grid:
        x = np.copy(inputs)
        x[:, feature_idx] = grid_point
        mean_output = np.mean(mlp(x), dtype=float)
        pdvalues.append(mean_output)
    return pdvalues


def get_partial_dependence_plots(
    mlp: MLP,
    inputs: ArrayLike,  # (n_samples, n_features)
    features_grid: Sequence[Sequence[float]],  # [...n_features][...n_grid_points]
) -> list[list[float]]:  # [...n_features][...n_grid_points]
    """
    Compute partial dependence values for each feature over a specified grid.

    Parameters
    ----------
    mlp : MLP
        An instance of the MLP class representing the multi-layer perceptron model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    features_grid : Sequence[Sequence[float]]
        A list where each element is a list of grid points for the corresponding feature.

    Returns
    -------
    list[list[float]]
        A list of lists containing the partial dependence values for each feature
        over the specified grid points.
    """
    return [
        get_partial_dependence_plot(
            mlp=mlp, inputs=inputs, feature_idx=feature_idx, grid=grid
        )
        for feature_idx, grid in enumerate(features_grid)
    ]


def get_partial_dependence_plot_distribution(
    mlps: MLPEnsemble,
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Sequence[float],  # [...n_grid_points]
) -> list[list[float]]:  # [...n_models][...n_grid_points]
    """
    Compute partial dependence values for a single feature over a specified grid
    for each model in the MLP ensemble.

    Parameters
    ----------
    mlps : MLPEnsemble
        An instance of the MLPEnsemble class representing the ensemble of MLP models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Sequence[float]
        A list of grid points for the specified feature.

    Returns
    -------
    list[list[float]]
        A list where each element is a list of partial dependence values for the specified
        feature over the given grid points for each model in the ensemble.
    """
    return [
        get_partial_dependence_plot(
            mlp=mlp, inputs=inputs, feature_idx=feature_idx, grid=grid
        )
        for mlp in mlps
    ]


def get_partial_dependence_plots_distribution(
    mlps: MLPEnsemble,
    inputs: ArrayLike,  # (n_samples, n_features)
    features_grid: Sequence[Sequence[float]],  # [...n_features][...n_grid_points]
) -> list[list[list[float]]]:  # [...n_features][...n_models][...n_grid_points]
    """
    Compute partial dependence values for each feature over a specified grid
    for each model in the MLP ensemble.

    Parameters
    ----------
    mlps : MLPEnsemble
        An instance of the MLPEnsemble class representing the ensemble of MLP models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    features_grid : Sequence[Sequence[float]]
        A list where each element is a list of grid points for the corresponding feature.

    Returns
    -------
    list[list[list[float]]]
        A list where each element corresponds to a feature and contains a list of lists,
        each inner list containing the partial dependence values over the specified grid points
        for each model in the ensemble.
    """
    return [
        get_partial_dependence_plot_distribution(
            mlps=mlps, inputs=inputs, feature_idx=feature_idx, grid=grid
        )
        for feature_idx, grid in enumerate(features_grid)
    ]


def get_median_partial_dependence_plot(
    mlps: MLPEnsemble,
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Sequence[float],  # [...n_grid_points]
) -> list[float]:  # [...n_grid_points]
    """
    Compute median partial dependence values for a single feature over a specified grid
    across the ensemble of MLP models.

    Parameters
    ----------
    mlps : MLPEnsemble
        An instance of the MLPEnsemble class representing the ensemble of MLP models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Sequence[float]
        A list of grid points for the specified feature.

    Returns
    -------
    list[float]
        A list containing the median partial dependence values for the specified feature
        over the given grid points across the ensemble.
    """
    pdp_distribution = get_partial_dependence_plot_distribution(
        mlps=mlps, inputs=inputs, feature_idx=feature_idx, grid=grid
    )  # [...n_models][...n_grid_points]
    return list(np.median(pdp_distribution, axis=0))  # [...n_grid_points]


def get_median_partial_dependence_plot_distribution(
    mlp_ensembles: list[MLPEnsemble],
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Sequence[float],  # [...n_grid_points]
    n_jobs: int = -1,
) -> list[list[float]]:  # [...n_ensembles][...n_grid_points]
    """
    Compute median partial dependence values for a single feature over a specified grid
    across multiple ensembles of MLP models.

    Parameters
    ----------
    ensembles : list[MLPEnsemble]
        A list of MLPEnsemble instances representing multiple ensembles of MLP models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Sequence[float]
        A list of grid points for the specified feature.
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1, which uses all available processors.

    Returns
    -------
    list[list[float]]
        A list where each element is a list containing the median partial dependence values
        for the specified feature over the given grid points for each ensemble.
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(get_median_partial_dependence_plot)(
            mlps=mlps,
            inputs=inputs,
            feature_idx=feature_idx,
            grid=grid,
        )
        for mlps in tqdm(mlp_ensembles, desc="Computing median PDPs")
    )

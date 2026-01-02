from collections.abc import Iterable

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from bella_companion.backend.type_hints import Array, Model


def get_partial_dependence_plot(
    model: Model,
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Iterable[float],  # [n_grid_points]
) -> list[float]:  # [n_grid_points]
    """
    Compute partial dependence values for a single feature over a specified grid.

    Parameters
    ----------
    model : Model
        A callable representing the model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Iterable[float]
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
        mean_output = np.mean(model(x), dtype=float)
        pdvalues.append(mean_output)
    return pdvalues


def get_partial_dependence_plots(
    model: Model,
    inputs: ArrayLike,  # (n_samples, n_features)
    features_grid: Iterable[Iterable[float]],  # [n_features][n_grid_points]
) -> list[list[float]]:  # [n_features][n_grid_points]
    """
    Compute partial dependence values for each feature over a specified grid.

    Parameters
    ----------
    model : Model
        A callable representing the model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    features_grid : Iterable[Iterable[float]]
        A list where each element is a list of grid points for the corresponding feature.

    Returns
    -------
    list[list[float]]
        A list of lists containing the partial dependence values for each feature
        over the specified grid points.
    """
    return [
        get_partial_dependence_plot(
            model=model, inputs=inputs, feature_idx=feature_idx, grid=grid
        )
        for feature_idx, grid in enumerate(features_grid)
    ]


def get_partial_dependence_plot_distribution(
    models: Iterable[Model],  # [n_models]
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Iterable[float],  # [n_grid_points]
) -> Array:  # (n_models, n_grid_points)
    """
    Compute partial dependence values for a single feature over a specified grid
    for each model in the ensemble.

    Parameters
    ----------
    models : Iterable[Model]
        An iterable of callables representing the ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Iterable[float]
        A list of grid points for the specified feature.

    Returns
    -------
    Array
        A 2D numpy array of shape (n_models, n_grid_points) containing the partial
        dependence values for the specified feature over the given grid points
        for each model in the ensemble.
    """
    return np.array(
        [
            get_partial_dependence_plot(
                model=model, inputs=inputs, feature_idx=feature_idx, grid=grid
            )
            for model in models
        ]
    )


def get_partial_dependence_plots_distribution(
    models: Iterable[Model],
    inputs: ArrayLike,  # (n_samples, n_features)
    features_grid: Iterable[Iterable[float]],  # [n_features][n_grid_points]
) -> list[Array]:  # [n_features](n_models, n_grid_points)
    """
    Compute partial dependence values for each feature over a specified grid
    for each model in the ensemble.

    Parameters
    ----------
    models : Iterable[Model]
        An iterable of callables representing the ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    features_grid : Iterable[Iterable[float]]
        A list where each element is a list of grid points for the corresponding feature.

    Returns
    -------
    list[Array]
        A list where each element is a 2D numpy array of shape (n_models, n_grid_points)
        containing the partial dependence values for the corresponding feature
        over the specified grid points for each model in the ensemble.
    """
    return [
        get_partial_dependence_plot_distribution(
            models=models, inputs=inputs, feature_idx=feature_idx, grid=grid
        )
        for feature_idx, grid in enumerate(features_grid)
    ]


def get_median_partial_dependence_plot(
    models: Iterable[Model],
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Iterable[float],  # [n_grid_points]
) -> Array:  # (n_grid_points, )
    """
    Compute median partial dependence values for a single feature over a specified grid
    across the ensemble of models.

    Parameters
    ----------
    models : Iterable[Model]
        An iterable of callables representing the ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Iterable[float]
        A list of grid points for the specified feature.

    Returns
    -------
    Array
        A 1D numpy array containing the median partial dependence values for the specified feature
        over the given grid points across the ensemble.
    """
    pdp_distribution = get_partial_dependence_plot_distribution(
        models=models, inputs=inputs, feature_idx=feature_idx, grid=grid
    )  # (n_models, n_grid_points)
    return np.median(pdp_distribution, axis=0)


def get_median_partial_dependence_plot_distribution(
    models: Iterable[Iterable[Model]],
    inputs: ArrayLike,  # (n_samples, n_features)
    feature_idx: int,
    grid: Iterable[float],  # [n_grid_points]
    n_jobs: int = -1,
) -> Array:  # (n_ensembles, n_grid_points)
    """
    Compute median partial dependence values for a single feature over a specified grid
    across multiple ensembles of models.

    Parameters
    ----------
    models : Iterable[Iterable[Model]]
        An iterable of iterables, where each inner iterable represents an ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    feature_idx : int
        The index of the feature for which to compute partial dependence.
    grid : Iterable[float]
        A list of grid points for the specified feature.
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1, which uses all available processors.

    Returns
    -------
    Array
        A 2D numpy array of shape (n_ensembles, n_grid_points) containing the median partial dependence values
        for the specified feature over the given grid points for each ensemble.
    """
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(get_median_partial_dependence_plot)(
                models=ensemble,
                inputs=inputs,
                feature_idx=feature_idx,
                grid=grid,
            )
            for ensemble in tqdm(models, desc="Computing median PDPs")
        )
    )

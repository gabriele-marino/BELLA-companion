from collections.abc import Iterable

import numpy as np
import shap  # pyright: ignore
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from bella_companion.backend.type_hints import Array, Model


def get_shap_values(
    model: Model,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_samples, n_features)
    """
    Compute SHAP feature importance values for the given inputs and model.

    Parameters
    ----------
    model : Model
        A callable representing the model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    background : ArrayLike | None, optional
        Background data for SHAP explainer of shape (n_background_samples, n_features).
        Default is None, in which case the inputs are used as background.

    Returns
    -------
    Array
        SHAP values for the inputs, of shape (n_samples, n_features).
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    if background is None:
        background = inputs
    explainer = shap.Explainer(model, background)
    return explainer(inputs).values  # pyright: ignore


def get_shap_features_importance(
    model: Model,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_features,)
    """
    Compute SHAP feature importance values for the given inputs and model.

    Parameters
    ----------
    model : Model
        A callable representing the model.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    background : ArrayLike | None, optional
        Background data for SHAP explainer of shape (n_background_samples, n_features).
        Default is None, in which case the inputs are used as background.

    Returns
    -------
    Array
        SHAP feature importance values for the inputs, of shape (n_features,).
    """
    shap_values = get_shap_values(model, inputs, background)
    return np.mean(np.abs(shap_values), axis=0)


def get_shap_feature_importance_distribution(
    models: Iterable[Model],
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_models, n_features)
    """
    Compute SHAP feature importance values for an ensemble of models.

    Parameters
    ----------
    models : Iterable[Model]
        An iterable of callables representing the ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    background : ArrayLike | None, optional
        Background data for SHAP explainer of shape (n_background_samples, n_features).
        Default is None, in which case the inputs are used as background.

    Returns
    -------
    Array
        SHAP feature importance values for each model in the ensemble,
        of shape (n_models, n_features).
    """
    return np.array(
        [get_shap_features_importance(model, inputs, background) for model in models]
    )


def get_median_shap_feature_importance(
    models: Iterable[Model],
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_features)
    """
    Compute median SHAP feature importance values for an ensemble of models.

    Parameters
    ----------
    models : Iterable[Model]
        An iterable of callables representing the ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    background : ArrayLike | None, optional
        Background data for SHAP explainer of shape (n_background_samples, n_features).
        Default is None, in which case the inputs are used as background.

    Returns
    -------
    Array
        Median SHAP feature importance values across the ensemble,
        of shape (n_features,).
    """
    return np.median(
        get_shap_feature_importance_distribution(models, inputs, background), axis=0
    )


def get_median_shap_feature_importance_distribution(
    models: Iterable[Iterable[Model]],
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features
    n_jobs: int = -1,
) -> Array:  # (n_ensembles, n_features)
    """
    Compute median SHAP feature importance values for multiple ensembles of models.

    Parameters
    ----------
    models : Iterable[Iterable[Model]]
        An iterable of iterables, where each inner iterable represents an ensemble of models.
    inputs : ArrayLike
        Input data of shape (n_samples, n_features).
    background : ArrayLike | None, optional
        Background data for SHAP explainer of shape (n_background_samples, n_features).
        Default is None, in which case the inputs are used as background.
    n_jobs : int, optional
        The number of parallel jobs to run. Default is -1 (use all available cores).

    Returns
    -------
    Array
        Median SHAP feature importance values for each ensemble,
        of shape (n_ensembles, n_features).
    """
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(get_median_shap_feature_importance)(
                models=ensemble, inputs=inputs, background=background
            )
            for ensemble in tqdm(
                models,
                desc="Computing median SHAP feature importance for ensembles",
            )
        )
    )

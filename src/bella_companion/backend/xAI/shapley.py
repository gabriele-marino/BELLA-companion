import numpy as np
import shap  # pyright: ignore
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from bella_companion.backend.mlp import MLP, MLPEnsemble
from bella_companion.backend.type_hints import Array


def get_shap_values(
    mlp: MLP,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_samples, n_features)
    """
    Compute SHAP feature importance values for the given inputs and MLP weights.

    Parameters
    ----------
    mlp : MLP
        An instance of the MLP class representing the multi-layer perceptron model.
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
    explainer = shap.Explainer(mlp, background)
    return explainer(inputs).values  # pyright: ignore


def get_shap_features_importance(
    mlp: MLP,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_features,)
    """
    Compute SHAP feature importance values for the given inputs and MLP weights.

    Parameters
    ----------
    mlp : MLP
        An instance of the MLP class representing the multi-layer perceptron model.
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
    shap_values = get_shap_values(mlp, inputs, background)
    return np.mean(np.abs(shap_values), axis=0)


def get_shap_feature_importance_distribution(
    mlps: MLPEnsemble,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_models, n_features)
    """
    Compute SHAP feature importance values for an ensemble of MLPs.

    Parameters
    ----------
    mlps : MLPEnsemble
        An instance of the MLPEnsemble class representing the ensemble of MLP models.
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
        [get_shap_features_importance(mlp, inputs, background) for mlp in mlps]
    )


def get_median_shap_feature_importance(
    mlps: MLPEnsemble,
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features)
) -> Array:  # (n_features)
    """
    Compute median SHAP feature importance values for an ensemble of MLPs.

    Parameters
    ----------
    mlps : MLPEnsemble
        An instance of the MLPEnsemble class representing the ensemble of MLP models.
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
        get_shap_feature_importance_distribution(mlps, inputs, background), axis=0
    )


def get_median_shap_feature_importance_distribution(
    mlp_ensembles: list[MLPEnsemble],
    inputs: ArrayLike,  # (n_samples, n_features)
    background: ArrayLike | None = None,  # (n_background_samples, n_features
    n_jobs: int = -1,
) -> Array:  # (n_ensembles, n_features)
    """
    Compute median SHAP feature importance values for multiple ensembles of MLPs.

    Parameters
    ----------
    ensembles : list[MLPEnsemble]
        A list of MLPEnsemble instances representing multiple ensembles of MLP models.
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
                mlps=mlps, inputs=inputs, background=background
            )
            for mlps in tqdm(
                mlp_ensembles,
                desc="Computing median SHAP feature importance for ensembles",
            )
        )
    )

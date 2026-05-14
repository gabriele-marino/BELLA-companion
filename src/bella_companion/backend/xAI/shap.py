from collections.abc import Iterable

import numpy as np
import shap  # pyright: ignore
from jaxtyping import Float
from joblib import Parallel, delayed
from tqdm import tqdm

from bella_companion.settings import memory
from bella_companion.typings import Array, EnsembleModel, Model, PredictionInput


def shap_importance(
    model: Model,
    inputs: PredictionInput,
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
) -> Float[Array, "n_features"]:  # noqa: F821
    """Compute SHAP feature importance values for a given model and input data."""
    if background is None:
        background = inputs
    explainer = shap.Explainer(model, background)
    shap_values = explainer(inputs).values  # pyright: ignore
    return np.mean(np.abs(shap_values), axis=0)  # pyright: ignore


@memory.cache
def posterior_shap_importance(
    ensemble: EnsembleModel,
    inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
) -> Float[Array, "n_samples n_features"]:  # noqa: F722
    """Compute SHAP feature importance values for each model in the ensemble."""
    return np.array([shap_importance(model, inputs, background) for model in ensemble])


@memory.cache
def posterior_median_shap_importance(
    ensembles: Iterable[EnsembleModel],
    inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
    n_jobs: int = -1,
) -> Float[Array, "n_bayes_models n_features"]:  # noqa: F722
    """Compute the distribution of median posterior SHAP feature importance values for a given set of model ensembles."""

    def median_posterior_shap_importance(
        ensemble: EnsembleModel,
    ) -> Float[Array, "n_features"]:  # noqa: F821
        return np.median(
            posterior_shap_importance(ensemble, inputs, background), axis=0
        )

    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(median_posterior_shap_importance)(ensemble=ensemble)
            for ensemble in tqdm(
                ensembles,
                desc="Computing median posterior SHAP importance distribution",
            )
        )
    )

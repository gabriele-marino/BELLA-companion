from collections.abc import Iterable

import numpy as np
import shap  # pyright: ignore
from jaxtyping import Float
from joblib import Parallel, delayed
from tqdm import tqdm

from bella_companion.backend.mlp import MLP, BayesMLP
from bella_companion.settings import memory
from bella_companion.typings import Array


def shap_importance(
    mlp: MLP,
    inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
) -> Float[Array, "n_features"]:  # noqa: F821
    """Compute SHAP feature importance values for a given MLP model and input data."""
    if background is None:
        background = inputs
    explainer = shap.Explainer(mlp, background)
    shap_values = explainer(inputs).values  # pyright: ignore
    return np.mean(np.abs(shap_values), axis=0)  # pyright: ignore


@memory.cache
def posterior_shap_importance(
    bayes_mlp: BayesMLP,
    inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
) -> Float[Array, "n_samples n_features"]:  # noqa: F722
    """Compute SHAP feature importance values for each sampled MLP model in a Bayesian MLP."""
    return np.array([shap_importance(mlp, inputs, background) for mlp in bayes_mlp])


@memory.cache
def posterior_median_shap_importance(
    bayes_mlps: Iterable[BayesMLP],
    inputs: Float[Array, "batch_size n_features"],  # noqa: F722
    background: Float[Array, "background_size n_features"] | None = None,  # noqa: F722
    n_jobs: int = -1,
) -> Float[Array, "n_bayes_mlps n_features"]:  # noqa: F722
    """Compute the distribution of median posterior SHAP feature importance values for a given set of Bayesian MLPs."""

    def median_posterior_shap_importance(
        bayes_mlp: BayesMLP,
    ) -> Float[Array, "n_features"]:  # noqa: F821
        return np.median(
            posterior_shap_importance(bayes_mlp, inputs, background), axis=0
        )

    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(median_posterior_shap_importance)(bayes_mlp=bayes_mlp)
            for bayes_mlp in tqdm(
                bayes_mlps,
                desc="Computing median posterior SHAP importance distribution",
            )
        )
    )

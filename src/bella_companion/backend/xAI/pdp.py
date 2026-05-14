from collections.abc import Iterable

import numpy as np
from jaxtyping import Float
from joblib import Parallel, delayed
from tqdm import tqdm

from bella_companion.settings import memory
from bella_companion.typings import Array, BayesModel, PredictionInput


@memory.cache
def posterior_pdp(
    bayes_model: BayesModel,
    inputs: PredictionInput,
    feature_idx: int,
    grid: Float[Array, "n_grid_points"],  # noqa: F821
) -> Float[Array, "n_samples n_grid_points"]:  # noqa:F722
    """Compute the posterior partial dependence plot (PDP) values for a given feature across all sampled models."""
    pdvalues: list[float] = []
    for grid_point in grid:
        x = np.copy(inputs)
        x[:, feature_idx] = grid_point
        mean_output = np.mean(bayes_model(x), axis=1, dtype=float)
        pdvalues.append(mean_output)
    return np.array(pdvalues).T


@memory.cache
def posterior_median_pdp(
    bayes_models: Iterable[BayesModel],
    inputs: PredictionInput,
    feature_idx: int,
    grid: Float[Array, "n_grid_points"],  # noqa: F821
    n_jobs: int = -1,
) -> Float[Array, "n_bayes_models n_grid_points"]:  # noqa: F722
    """Compute the distribution of median partial dependence plot (PDP) values for a given feature across a set of Bayesian models."""

    def median_pdp(bayes_model: BayesModel) -> Float[Array, "n_grid_points"]:  # noqa: F821
        return np.median(
            posterior_pdp(
                bayes_model=bayes_model,
                inputs=inputs,
                feature_idx=feature_idx,
                grid=grid,
            ),
            axis=0,
        )

    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(median_pdp)(bayes_model=bayes_model)
            for bayes_model in tqdm(
                bayes_models, desc="Computing median posterior PDP distribution"
            )
        )
    )

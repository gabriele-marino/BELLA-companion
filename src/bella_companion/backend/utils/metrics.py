import numpy as np
import pandas as pd

from bella_companion.backend.utils.beast import (
    ESS_POSTFIX,
    LOWER_POSTFIX,
    MEDIAN_POSTFIX,
    UPPER_POSTFIX,
)
from bella_companion.backend.utils.slurm import TOTAL_HOURS_KEY


def mae_distribution_from_summaries(
    summaries: pd.DataFrame, true_values: dict[str, float]
) -> list[float]:
    """Compute the Mean Absolute Error (MAE) for each MCMC run.

    Let theta = (theta_1, ..., theta_p) be the vector of true parameter values,
    and let theta_hat^(i) = (theta_hat_1^(i), ..., theta_hat_p^(i)) be the vector
    of posterior median estimates for run i. The MAE for run i is defined as

        MAE_i = (1/p) * sum_{j=1}^p |theta_hat_j^(i) - theta_j|,

    where p is the number of parameters.

    Parameters
    ----------
    summaries : pandas.DataFrame
        DataFrame with one row per MCMC run. Columns must include posterior
        medians for each parameter, named as "{parameter_name}{MEDIAN_POSTFIX}".
    true_values : dict of str to float
        Mapping from parameter names to their true values.

    Returns
    -------
    list of float
        List of MAE values, one per MCMC run.
    """
    medians = summaries[[f"{t}{MEDIAN_POSTFIX}" for t in true_values]].values
    true_vec = np.array(list(true_values.values()))
    return np.abs(medians - true_vec).mean(axis=1).tolist()


def mae_from_summaries(summaries: pd.DataFrame, true_values: dict[str, float]) -> float:
    """Compute the Mean Absolute Error (MAE) using aggregated posterior medians.

    Let theta = (theta_1, ..., theta_p) be the vector of true parameter values.
    For each run i and parameter j, let theta_hat_j^(i) denote the posterior
    median estimate. Define the aggregated estimate for parameter j as the
    median across runs:

        theta_tilde_j = median({theta_hat_j^(i) : i = 1, ..., n}),

    where n is the number of MCMC runs.

    The MAE is then

        MAE = (1/p) * sum_{j=1}^p |theta_tilde_j - theta_j|,

    where p is the number of parameters.

    Parameters
    ----------
    summaries : pandas.DataFrame
        DataFrame with one row per MCMC run. Columns must include posterior
        medians for each parameter, named as "{parameter_name}{MEDIAN_POSTFIX}".
    true_values : dict of str to float
        Mapping from parameter names to their true values.

    Returns
    -------
    float
        Mean Absolute Error between aggregated posterior medians and true values.
    """
    median_columns = [f"{t}{MEDIAN_POSTFIX}" for t in true_values]
    preds = summaries[median_columns].median(axis=0).values
    targets = np.array(list(true_values.values()))
    return np.mean(np.abs(preds - targets), dtype=float)


def mse_from_summaries(summaries: pd.DataFrame, true_values: dict[str, float]) -> float:
    """Compute the Mean Squared Error (MSE) using aggregated posterior medians.

    Let theta = (theta_1, ..., theta_p) be the vector of true parameter values.
    For each run i and parameter j, let theta_hat_j^(i) denote the posterior
    median estimate. Define the aggregated estimate for parameter j as the
    median across runs:

        theta_tilde_j = median({theta_hat_j^(i) : i = 1, ..., n}),

    where n is the number of MCMC runs.

    The MSE is then

        MSE = (1/p) * sum_{j=1}^p (theta_tilde_j - theta_j)^2,

    where p is the number of parameters.

    Parameters
    ----------
    summaries : pandas.DataFrame
        DataFrame with one row per MCMC run. Columns must include posterior
        medians for each parameter, named as "{parameter_name}{MEDIAN_POSTFIX}".
    true_values : dict of str to float
        Mapping from parameter names to their true values.

    Returns
    -------
    float
        Mean Squared Error between aggregated posterior medians and true values.
    """
    median_columns = [f"{t}{MEDIAN_POSTFIX}" for t in true_values]
    preds = summaries[median_columns].median(axis=0).values
    targets = np.array(list(true_values.values()))
    return np.mean((preds - targets) ** 2, dtype=float)


def coverage_from_summaries(
    summaries: pd.DataFrame, true_values: dict[str, float]
) -> float:
    """Compute the average empirical coverage probability across parameters.

    For each parameter j, let theta_j be the true value, and for each run i let
    [L_j^(i), U_j^(i)] denote the posterior interval (e.g., a credible interval)
    given by the corresponding lower and upper summary statistics.

    The empirical coverage for parameter j is

        C_j = (1/n) * sum_{i=1}^n 1{ L_j^(i) <= theta_j <= U_j^(i) },

    where n is the number of MCMC runs and 1{·} is the indicator function.

    The function returns the average coverage across all parameters:

        C = (1/p) * sum_{j=1}^p C_j,

    where p is the number of parameters.

    Parameters
    ----------
    summaries : pandas.DataFrame
        DataFrame with one row per MCMC run. Columns must include lower and
        upper bounds for each parameter, named as "{parameter_name}{LOWER_POSTFIX}"
        and "{parameter_name}{UPPER_POSTFIX}" respectively.
    true_values : dict of str to float
        Mapping from parameter names to their true values.

    Returns
    -------
    float
        Mean empirical coverage probability across parameters.
    """
    coverages = [
        (
            (summaries[f"{target}{LOWER_POSTFIX}"] <= true_values[target])
            & (true_values[target] <= summaries[f"{target}{UPPER_POSTFIX}"])
        ).mean()
        for target in true_values
    ]
    return np.mean(coverages, dtype=float)


def avg_ci_width_from_summaries(summaries: pd.DataFrame, targets: list[str]) -> float:
    widths = [
        np.mean(
            summaries[f"{target}{UPPER_POSTFIX}"]
            - summaries[f"{target}{LOWER_POSTFIX}"]
        )
        for target in targets
    ]
    return np.mean(widths, dtype=float)


def mean_ess_per_hour_from_summaries(
    summaries: pd.DataFrame, targets: list[str]
) -> float:
    ess_cols = [f"{t}{ESS_POSTFIX}" for t in targets]
    mean_ess_per_hour = summaries[ess_cols].mean(axis=1) / summaries[TOTAL_HOURS_KEY]
    return mean_ess_per_hour.mean()

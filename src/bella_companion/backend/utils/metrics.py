import numpy as np
import pandas as pd

from bella_companion.backend.type_hints import Array
from bella_companion.backend.utils.beast import (
    ESS_POSTFIX,
    LOWER_POSTFIX,
    MEDIAN_POSTFIX,
    UPPER_POSTFIX,
)
from bella_companion.backend.utils.slurm import TOTAL_HOURS_KEY


def mae_distribution_from_summaries(
    summaries: pd.DataFrame, true_values: dict[str, float]
) -> Array:
    """
    Computes the MAE distribution between median estimates and true values
    for each row in the summaries DataFrame.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing BEAST summaries with median estimates.
    true_values : dict[str, float]
        Dictionary of true target values.

    Returns
    -------
    Array
        Array of MAE values for each row in the summaries DataFrame.
    """
    medians = summaries[[f"{t}{MEDIAN_POSTFIX}" for t in true_values]].values
    true_vec = np.array(list(true_values.values()))
    return np.abs(medians - true_vec).mean(axis=1)


def mae_from_summaries(summaries: pd.DataFrame, true_values: dict[str, float]) -> float:
    """
    Computes the mean MAE between median estimates and true values
    from BEAST summaries DataFrame.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing BEAST summaries with median estimates.
    true_values : dict[str, float]
        Dictionary of true target values.

    Returns
    -------
    float
        The mean MAE value.
    """
    median_columns = [f"{t}{MEDIAN_POSTFIX}" for t in true_values]
    preds = summaries[median_columns].median(axis=0).values
    targets = np.array(list(true_values.values()))
    return np.mean(np.abs(preds - targets), dtype=float)


def coverage_from_summaries(
    summaries: pd.DataFrame, true_values: dict[str, float]
) -> float:
    """
    Computes the coverage of the highest density intervals (HDI) from BEAST summaries DataFrame.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing BEAST summaries with lower and upper bounds.
    true_values : dict[str, float]
        Dictionary of true target values.

    Returns
    -------
    float
        The coverage proportion of true values within the HDI intervals.
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
    """
    Computes the average width of the highest density intervals (HDI) from BEAST summaries DataFrame.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing BEAST summaries with lower and upper bounds.
    targets : list[str]
        List of target names to consider for interval width calculation.

    Returns
    -------
    float
        The average width of the HDI intervals.
    """
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
    """
    Computes the mean ESS per hour from BEAST summaries DataFrame.

    Parameters
    ----------
    summaries : pd.DataFrame
        DataFrame containing BEAST summaries with ESS values and total computation hours.
    targets : list[str]
        List of target names to consider for ESS calculation.

    Returns
    -------
    float
        The mean ESS per hour across the specified targets.
    """
    ess_cols = [f"{t}{ESS_POSTFIX}" for t in targets]
    mean_ess_per_hour = summaries[ess_cols].mean(axis=1) / summaries[TOTAL_HOURS_KEY]
    return mean_ess_per_hour.mean()

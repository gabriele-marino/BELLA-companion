import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from statsmodels.nonparametric.smoothers_lowess import lowess  # pyright: ignore

from bella_companion.backend.plots import Color
from bella_companion.simulations.scenarios import EPI_MULTITYPE


def plot_estimates(ax: Axes, summaries: pd.DataFrame, color: Color):
    (migration_rate,) = EPI_MULTITYPE.targets

    sort_idx = np.argsort(EPI_MULTITYPE.migration_predictor.flatten())
    predictors = EPI_MULTITYPE.migration_predictor.flatten()[sort_idx]
    true_rates = EPI_MULTITYPE.migration_rates.flatten()[sort_idx]

    estimates = np.array(
        [summaries[f"{key}.median"].median() for key in migration_rate.keys]
    )[sort_idx]
    lower = np.array(
        [summaries[f"{key}.lower"].median() for key in migration_rate.keys]
    )[sort_idx]
    upper = np.array(
        [summaries[f"{key}.upper"].median() for key in migration_rate.keys]
    )[sort_idx]

    ax.errorbar(  # pyright: ignore
        predictors,
        estimates,
        yerr=[estimates - lower, upper - estimates],
        fmt="o",
        color=color,
        elinewidth=2,
        capsize=5,
        linewidth=3,
    )

    x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
    lowess_fit = lowess(np.log(estimates + 1e-8), predictors, frac=0.4)  # pyright: ignore
    y_smooth = np.exp(np.interp(x_smooth, lowess_fit[:, 0], lowess_fit[:, 1]))  # pyright: ignore

    ax.plot(  # pyright: ignore
        x_smooth,
        y_smooth,
        color=color,
        linestyle="-",
        alpha=0.7,
        linewidth=3,
    )

    ax.plot(predictors, true_rates, linestyle="--", marker="o", color="k", linewidth=3)  # pyright: ignore

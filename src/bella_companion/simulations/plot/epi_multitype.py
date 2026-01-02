import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from bella_companion.backend.utils.beast import (
    LOWER_POSTFIX,
    MEDIAN_POSTFIX,
    UPPER_POSTFIX,
)
from bella_companion.simulations.plot.globals import COLORS
from bella_companion.simulations.scenarios.epi_multitype import (
    EPI_MULTITYPE_SCENARIO,
    MIGRATION_PREDICTOR,
    MIGRATION_RATES,
)


def plot_epi_multitype():
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "epi-multitype"
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "epi-multitype"
    os.makedirs(output_dir, exist_ok=True)

    models_summaries = {
        model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for model in ["PA", "GLM", "BELLA-32_16"]
    }

    sort_idx = np.argsort(MIGRATION_PREDICTOR.flatten())
    predictors = MIGRATION_PREDICTOR.flatten()[sort_idx]
    true_rates = MIGRATION_RATES.flatten()[sort_idx]

    targets = EPI_MULTITYPE_SCENARIO.targets["migrationRate"]
    for model, summaries in models_summaries.items():
        estimates = np.array(
            [summaries[f"{target}{MEDIAN_POSTFIX}"].median() for target in targets]
        )[sort_idx]
        lower = np.array(
            [summaries[f"{target}{LOWER_POSTFIX}"].median() for target in targets]
        )[sort_idx]
        upper = np.array(
            [summaries[f"{target}{UPPER_POSTFIX}"].median() for target in targets]
        )[sort_idx]

        plt.errorbar(  # pyright: ignore
            predictors,
            estimates,
            yerr=[estimates - lower, upper - estimates],
            fmt="o",
            color=COLORS[model],
            elinewidth=2,
            capsize=5,
        )

        spline = UnivariateSpline(predictors, estimates)
        x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, color=COLORS[model], linestyle="-", alpha=0.7)  # pyright: ignore

        plt.plot(  # pyright: ignore
            predictors, true_rates, linestyle="--", marker="o", color="k"
        )

        plt.xlabel("Migration predictor")  # pyright: ignore
        plt.ylabel("Migration rate")  # pyright: ignore
        plt.savefig(output_dir / f"{model}-predictions.svg")  # pyright: ignore
        plt.close()

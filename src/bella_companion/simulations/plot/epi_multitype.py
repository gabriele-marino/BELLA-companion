import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess  # pyright: ignore

from bella_companion.backend.utils.beast import (
    LOWER_POSTFIX,
    MEDIAN_POSTFIX,
    UPPER_POSTFIX,
)
from bella_companion.settings import BELLA_REFERENCE_MODEL, BELLA_SETTINGS, MODEL_COLORS
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
        for model in ["PA", "GLM", BELLA_REFERENCE_MODEL]
    }
    bella_models_summaries = {
        model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for model in BELLA_SETTINGS
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
            color=MODEL_COLORS[model],
            elinewidth=2,
            capsize=5,
        )

        x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
        lowess_fit = lowess(np.log(estimates + 1e-8), predictors, frac=0.4)  # pyright: ignore
        y_smooth = np.exp(np.interp(x_smooth, lowess_fit[:, 0], lowess_fit[:, 1]))  # pyright: ignore

        plt.plot(  # pyright: ignore
            x_smooth, y_smooth, color=MODEL_COLORS[model], linestyle="-", alpha=0.7
        )

        plt.plot(  # pyright: ignore
            predictors, true_rates, linestyle="--", marker="o", color="k"
        )

        plt.xlabel("Migration predictor")  # pyright: ignore
        plt.ylabel("Migration rate")  # pyright: ignore
        plt.savefig(output_dir / f"{model}-predictions.svg")  # pyright: ignore
        plt.close()

    for filename, variance_summaries in [
        ("splines-comparison", models_summaries),
        ("bella-variance", bella_models_summaries),
    ]:
        for model, summaries in variance_summaries.items():
            estimates = np.array(
                [summaries[f"{target}{MEDIAN_POSTFIX}"].median() for target in targets]
            )[sort_idx]

            x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
            lowess_fit = lowess(np.log(estimates + 1e-8), predictors, frac=0.4)  # pyright: ignore
            y_smooth = np.exp(np.interp(x_smooth, lowess_fit[:, 0], lowess_fit[:, 1]))  # pyright: ignore
            plt.scatter(  # pyright: ignore
                predictors, estimates, color=MODEL_COLORS[model], label=model
            )
            plt.plot(  # pyright: ignore
                x_smooth, y_smooth, color=MODEL_COLORS[model], linestyle="-", alpha=0.7
            )

        plt.plot(  # pyright: ignore
            predictors, true_rates, linestyle="--", marker="o", color="k"
        )
        plt.xlabel("Migration predictor")  # pyright: ignore
        plt.ylabel("Migration rate")  # pyright: ignore
        plt.ylim((None, 0.13))  # pyright: ignore
        plt.savefig(output_dir / f"{filename}.svg")  # pyright: ignore
        plt.close()

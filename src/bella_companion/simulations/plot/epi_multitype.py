import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from bella_companion.simulations.scenarios.epi_multitype import (
    MIGRATION_PREDICTOR,
    MIGRATION_RATES,
    SCENARIO,
)


def plot_predictions(output_dir: Path):
    summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / "epi-multitype"

    log_summaries = {
        model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for model in ["PA", "GLM", "BELLA-32_16"]
    }

    sort_idx = np.argsort(MIGRATION_PREDICTOR.flatten())
    predictors = MIGRATION_PREDICTOR.flatten()[sort_idx]
    true_rates = MIGRATION_RATES.flatten()[sort_idx]

    targets = SCENARIO.targets["migrationRate"]
    for i, (model, log_summary) in enumerate(log_summaries.items()):
        estimates = np.array(
            [log_summary[f"{target}_median"].median() for target in targets]
        )[sort_idx]
        lower = np.array(
            [log_summary[f"{target}_lower"].median() for target in targets]
        )[sort_idx]
        upper = np.array(
            [log_summary[f"{target}_upper"].median() for target in targets]
        )[sort_idx]

        plt.errorbar(  # pyright: ignore
            predictors,
            estimates,
            yerr=[estimates - lower, upper - estimates],
            fmt="o",
            color=f"C{i}",
            elinewidth=2,
            capsize=5,
        )

        spline = UnivariateSpline(predictors, estimates)
        x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, color=f"C{i}", linestyle="-", alpha=0.7)  # pyright: ignore

        plt.plot(  # pyright: ignore
            predictors, true_rates, linestyle="--", marker="o", color="k"
        )

        plt.xlabel("Migration predictor")  # pyright: ignore
        plt.ylabel("Migration rate")  # pyright: ignore
        plt.savefig(output_dir / f"{model}-predictions.svg")  # pyright: ignore
        plt.close()

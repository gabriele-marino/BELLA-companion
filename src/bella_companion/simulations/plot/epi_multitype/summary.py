import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess  # pyright: ignore

from bella_companion.settings import settings
from bella_companion.simulations.scenarios.epi_multitype import EPI_MULTITYPE


def plot_summary():
    summaries_dir = settings.summaries_dir / "epi-multitype"
    output_dir = settings.figures_dir / "epi-multitype"
    os.makedirs(output_dir, exist_ok=True)

    reference_model = settings.bella_reference_models["epi-multitype"]
    models = ["PA", "GLM", reference_model]
    models_summaries = {
        model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for model in models
    }

    sort_idx = np.argsort(EPI_MULTITYPE.migration_predictor.flatten())
    predictors = EPI_MULTITYPE.migration_predictor.flatten()[sort_idx]
    true_rates = EPI_MULTITYPE.migration_rates.flatten()[sort_idx]

    (migration_rate,) = EPI_MULTITYPE.targets

    _, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained", sharey=True)  # pyright: ignore

    for ax, (model, summaries) in zip(axes, models_summaries.items()):
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
            color=settings.model_colors[model],
            elinewidth=2,
            capsize=5,
            linewidth=3,
        )

        x_smooth = np.linspace(np.min(predictors), np.max(predictors), 100)
        lowess_fit = lowess(np.log(estimates + 1e-8), predictors, frac=0.4)  # pyright: ignore
        y_smooth = np.exp(np.interp(x_smooth, lowess_fit[:, 0], lowess_fit[:, 1]))  # pyright: ignore

        ax.plot(
            x_smooth,
            y_smooth,
            color=settings.model_colors[model],
            linestyle="-",
            alpha=0.7,
            linewidth=3,
        )

        ax.plot(
            predictors, true_rates, linestyle="--", marker="o", color="k", linewidth=3
        )

        ax.set_xlabel("Migration predictor")

    for col, model in enumerate(models):
        handle = Patch(
            facecolor=settings.model_colors[model], edgecolor="none", label=model
        )
        axes[col].legend(
            handles=[handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            fontsize=14,
            handlelength=3,
            handleheight=1.5,
        )

    axes[0].set_ylabel("Migration rate")
    plt.savefig(output_dir / "summary.svg")  # pyright: ignore

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from bella_companion.settings import settings
from bella_companion.simulations.plot.epi_multitype.common import plot_estimates


def plot_summary(output_file: Path):
    summaries_dir = settings.summaries_dir / "epi-multitype"

    reference_model = settings.bella_reference_models["epi-multitype"]
    models = ["PA", "GLM", reference_model]
    models_summaries = {
        model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
        for model in models
    }

    _, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained", sharey=True)  # pyright: ignore

    for ax, (model, summaries) in zip(axes, models_summaries.items()):
        plot_estimates(ax=ax, summaries=summaries, color=settings.model_colors[model])
        ax.set_xlabel(r"$x_{ij}$")

    for col, model in enumerate(models):
        handle = Patch(
            facecolor=settings.model_colors[model],
            edgecolor="none",
            label=model if model != reference_model else "BELLA",
        )
        axes[col].legend(
            handles=[handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            fontsize=14,
            handlelength=3,
            handleheight=1.5,
        )
    axes[0].set_ylabel(r"$m_{ij}$")

    plt.savefig(output_file)  # pyright: ignore

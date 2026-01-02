import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bella_companion.backend import (
    MEDIAN_POSTFIX,
    coverage_from_summaries,
    mae_distribution_from_summaries,
    skyline_plot,
)
from bella_companion.simulations.plot.globals import COLORS
from bella_companion.simulations.scenarios.epi_skyline import REPRODUCTION_NUMBERS


def plot_epi_skyline():
    base_output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "epi-skyline"

    mlp_models = {1: "3_2", 2: "16_8", 3: "32_16"}
    for i, reproduction_number in enumerate(REPRODUCTION_NUMBERS, start=1):
        summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / f"epi-skyline_{i}"
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in ["PA", "GLM", f"BELLA-{mlp_models[i]}"]
        }

        output_dir = base_output_dir / str(i)
        os.makedirs(output_dir, exist_ok=True)

        for model, summaries in models_summaries.items():
            skyline_plot(
                [
                    summaries[f"reproductionNumberSPi{i}{MEDIAN_POSTFIX}"].median()
                    for i in range(len(reproduction_number))
                ],
                step_kwargs={"color": COLORS[model]},
            )
        skyline_plot(reproduction_number, step_kwargs={"color": "k", "linestyle": "--"})
        plt.xlabel("Time")  # pyright: ignore
        plt.ylabel(r"$R_t$")  # pyright: ignore
        plt.savefig(output_dir / "predictions.svg")  # pyright: ignore
        plt.close()

        for model, summaries in models_summaries.items():
            coverage_by_time_bin = [
                coverage_from_summaries(
                    summaries=summaries, true_values={f"reproductionNumberSPi{i}": R}
                )
                for i, R in enumerate(reproduction_number)
            ]
            plt.plot(coverage_by_time_bin, marker="o", color=COLORS[model])  # pyright: ignore
        plt.xlabel("Time bin")  # pyright: ignore
        plt.ylabel("Coverage")  # pyright: ignore
        plt.ylim((0, 1.05))  # pyright: ignore
        plt.savefig(output_dir / "coverage.svg")  # pyright: ignore
        plt.close()

        maes = pd.concat(
            [
                pd.DataFrame(
                    {
                        "MAE": mae_distribution_from_summaries(
                            summaries=summaries,
                            true_values={f"reproductionNumberSPi{i}": R},
                        )
                    }
                )
                .assign(Model=model)
                .assign(**{"Time bin": i})
                for model, summaries in models_summaries.items()
                for i, R in enumerate(reproduction_number)
            ]
        )
        sns.violinplot(
            x="Time bin",
            y="MAE",
            hue="Model",
            data=maes,
            inner=None,
            cut=0,
            density_norm="width",
            palette=COLORS,
            legend=False,
        )
        plt.savefig(output_dir / "maes.svg")  # pyright: ignore
        plt.close()

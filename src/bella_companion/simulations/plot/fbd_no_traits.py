import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bella_companion.simulations.plot.utils import (
    plot_coverage_per_time_bin,
    plot_maes_per_time_bin,
    step,
)
from bella_companion.simulations.scenarios.fbd_no_traits import RATES


def plot_fbd_no_traits_results():
    base_output = Path(os.environ["BELLA_FIGURES_DIR"]) / "fbd-no-traits"

    mlp_models = {1: "3_2", 2: "16_8", 3: "3_2"}
    for i, rates in enumerate(RATES, start=1):
        summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / f"fbd-no-traits_{i}"
        logs_summaries = {
            "Nonparametric": pd.read_csv(summaries_dir / "Nonparametric.csv"),
            "GLM": pd.read_csv(summaries_dir / "GLM.csv"),
            "MLP": pd.read_csv(summaries_dir / f"MLP-{mlp_models[i]}.csv"),
        }
        true_values = {"birthRateSP": rates["birth"], "deathRateSP": rates["death"]}

        output_dir = base_output / str(i)
        for id, rate in true_values.items():
            for log_summary in logs_summaries.values():
                step(
                    [
                        float(np.median(log_summary[f"{id}i{i}_median"]))
                        for i in range(len(rate))
                    ],
                    reverse_xticks=True,
                )
            step(rate, color="k", linestyle="--", reverse_xticks=True)
            plt.ylabel(  # pyright: ignore
                r"$\lambda$" if id == "birthRateSP" else r"$\mu$"
            )
            plt.savefig(output_dir / f"{id}-predictions.svg")  # pyright: ignore
            plt.close()

        plot_coverage_per_time_bin(
            logs_summaries,
            true_values,
            output_dir / "coverage.svg",
            reverse_xticks=True,
        )
        plot_maes_per_time_bin(
            logs_summaries,
            true_values,
            output_dir / "maes.svg",
            reverse_xticks=True,
        )

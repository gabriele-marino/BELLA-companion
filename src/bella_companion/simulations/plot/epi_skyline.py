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
from bella_companion.simulations.scenarios.epi_skyline import REPRODUCTION_NUMBERS


def plot_epi_skyline_results():
    base_output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "epi-skyline"

    mlp_models = {1: "3_2", 2: "16_8", 3: "32_16"}
    for i, reproduction_number in enumerate(REPRODUCTION_NUMBERS, start=1):
        summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / f"epi-skyline_{i}"
        logs_summaries = {
            "PA": pd.read_csv(summaries_dir / "PA.csv"),
            "GLM": pd.read_csv(summaries_dir / "GLM.csv"),
            "BELLA": pd.read_csv(summaries_dir / f"BELLA-{mlp_models[i]}.csv"),
        }
        true_values = {"reproductionNumberSP": reproduction_number}

        output_dir = base_output_dir / str(i)
        os.makedirs(output_dir, exist_ok=True)
        for log_summary in logs_summaries.values():
            step(
                [
                    np.median(log_summary[f"reproductionNumberSPi{i}_median"])
                    for i in range(len(reproduction_number))
                ]
            )
        step(reproduction_number, color="k", linestyle="--")
        plt.ylabel(r"R_0")  # pyright: ignore
        plt.savefig(output_dir / "predictions.svg")  # pyright: ignore
        plt.close()

        plot_coverage_per_time_bin(
            logs_summaries, true_values, output_dir / "coverage.svg"
        )
        plot_maes_per_time_bin(logs_summaries, true_values, output_dir / "maes.svg")

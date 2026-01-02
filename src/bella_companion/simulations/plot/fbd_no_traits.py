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
from bella_companion.simulations.scenarios.fbd_no_traits import RATES


def plot_fbd_no_traits():
    base_output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "fbd-no-traits"

    mlp_models = {1: "3_2", 2: "16_8", 3: "3_2"}
    for i, rates in enumerate(RATES, start=1):
        summaries_dir = Path(os.environ["BELLA_SUMMARIES_DIR"]) / f"fbd-no-traits_{i}"
        models_summaries = {
            model: pd.read_csv(summaries_dir / f"{model}.csv")  # pyright: ignore
            for model in [f"BELLA-{mlp_models[i]}", "GLM", "PA"]
        }

        for rate, values in rates.items():
            output_dir = base_output_dir / str(i) / rate
            os.makedirs(output_dir, exist_ok=True)

            for model, summaries in models_summaries.items():
                medians = [
                    summaries[f"{rate}RateSPi{i}{MEDIAN_POSTFIX}"].median()
                    for i in range(len(values))
                ]
                skyline_plot(
                    list(reversed(medians)), step_kwargs={"color": COLORS[model]}
                )
            skyline_plot(
                list(reversed(values)), step_kwargs={"color": "k", "linestyle": "--"}
            )
            plt.gca().invert_xaxis()
            plt.xlabel("Time")  # pyright: ignore
            plt.ylabel(  # pyright: ignore
                r"$\lambda$" if rate == "birth" else r"$\mu$"
            )
            plt.savefig(output_dir / "predictions.svg")  # pyright: ignore
            plt.close()

            for model, summaries in models_summaries.items():
                coverage_by_time_bin = [
                    coverage_from_summaries(
                        summaries=summaries, true_values={f"{rate}RateSPi{i}": v}
                    )
                    for i, v in enumerate(values)
                ]
                plt.plot(  # pyright: ignore
                    list(reversed(coverage_by_time_bin)),
                    marker="o",
                    color=COLORS[model],
                )
            plt.gca().invert_xaxis()
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
                                true_values={f"{rate}RateSPi{i}": v},
                            )
                        }
                    )
                    .assign(Model=model)
                    .assign(**{"Time bin": len(values) - i - 1})
                    for model, summaries in models_summaries.items()
                    for i, v in enumerate(values)
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
            plt.gca().invert_xaxis()
            plt.savefig(output_dir / "maes.svg")  # pyright: ignore
            plt.close()

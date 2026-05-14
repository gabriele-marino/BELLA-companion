from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bella_companion.backend.beast import read_log_file
from bella_companion.backend.mlp import BayesMLP
from bella_companion.backend.plots import ribbon_plot
from bella_companion.eucovid.settings import (
    BELLA_CONFIGS,
    PREDICTOR,
    PREDICTOR_TRANSFORMED,
)
from bella_companion.settings import settings


def plot_pdp(output_file: Path):
    summaries_dir = settings.summaries_dir / "eucovid"

    plt.figure(layout="constrained", figsize=(4, 3))  # pyright: ignore

    log = read_log_file(summaries_dir / "GLM" / "MCMC.combined.log", burn_in=0.0)
    log = log.sample(n=100, random_state=42)
    w = np.array(log["migrationRateW"])
    scaler = np.array(log["migrationRateScaler"])

    idx = np.argsort(PREDICTOR.flatten())
    input = PREDICTOR_TRANSFORMED.flatten()[idx]
    x = PREDICTOR.flatten()[idx]

    y = np.exp(np.log(scaler)[:, None] + np.outer(w, input))
    ribbon_plot(
        x=x,
        y=y,
        color="C1",
        label="GLM",
        show_samples=False,
    )

    bayes_mlp = BayesMLP.from_log_file(
        log_file=summaries_dir / "BELLA" / "MCMC.combined.log",
        id="migrationRate",
        hidden_activation=BELLA_CONFIGS.hidden_activation,
        output_activation="softplus",
        burn_in=0.0,
    )
    y = bayes_mlp(input.reshape(-1, 1))
    ribbon_plot(x=x, y=y, color="C2", label="BELLA", show_samples=False)

    plt.xlabel("Daily air passengers / 100,000 inhabitants")  # pyright: ignore
    plt.ylabel("Migration rate")  # pyright: ignore
    plt.xscale("log")  # pyright: ignore
    plt.yscale("log")  # pyright: ignore
    plt.legend()  # pyright: ignore
    plt.savefig(output_file)  # pyright: ignore

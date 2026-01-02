import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bella_companion.backend import (
    MLPEnsemble,
    get_partial_dependence_plot_distribution,
    normalize,
    read_log_file,
    ribbon_plot,
)
from bella_companion.eucovid.settings import DATA_DIR


def plot_eucovid_flights_over_population():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    logs_dir = (
        Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
        / "eucovid"
        / "flights_over_population"
    )

    log = read_log_file(logs_dir / "GLM" / "MCMC.combined.log", burn_in=0.0)
    log = log.sample(n=100, random_state=42)  # pyright: ignore
    w = np.array(log["migrationRateW"])
    scaler = np.array(log["migrationRateScaler"])
    data_dir = DATA_DIR / "flights_over_population"
    data = np.loadtxt(data_dir / "flights_over_population.csv")
    x = np.linspace(np.min(data), np.max(data), 10)
    y = np.exp(np.log(scaler)[:, None] + np.outer(w, x))
    ribbon_plot(
        x=normalize(x), y=y, color="C1", label="GLM", samples_kwargs={"linewidth": 1}
    )

    mlps = MLPEnsemble.from_log_file(
        log_file=logs_dir / "BELLA" / "MCMC.combined.log",
        target_name="migrationRate",
        hidden_activation="relu",
        output_activation="softplus",
        burn_in=0.0,
    )
    x = np.linspace(0, 1, 10)
    y = mlps(x.reshape(-1, 1))
    ribbon_plot(x=x, y=y, color="C2", label="BELLA", samples_kwargs={"linewidth": 1})

    plt.xlabel("N. Flights / Pop. Size")  # pyright: ignore
    plt.ylabel("Migration rate")  # pyright: ignore
    plt.yscale("log")  # pyright: ignore
    plt.legend()  # pyright: ignore
    plt.savefig(output_dir / "migration-rates-vs-flights-over-population.svg")  # pyright: ignore
    plt.close()


def plot_eucovid_flights_and_populations():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    logs_dir = (
        Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
        / "eucovid"
        / "flights_and_populations"
    )

    mlps = MLPEnsemble.from_log_file(
        log_file=logs_dir / "BELLA" / "MCMC.combined.log",
        target_name="migrationRate",
        hidden_activation="relu",
        output_activation="softplus",
        burn_in=0.0,
    )

    data_dir = DATA_DIR / "flights_and_populations"
    inputs = np.concat(
        [
            np.loadtxt(data_dir / f"{file}.csv").reshape(1, -1)
            for file in ["flights", "populations"]
        ]
    ).T

    for feature_idx, (feature, color) in enumerate(
        [
            ("N. Flights", "#56B4E9"),
            ("Pop. Size", "#009E73"),
        ]
    ):
        grid = np.linspace(0, 1, 10).tolist()
        pdps = get_partial_dependence_plot_distribution(
            models=mlps,
            inputs=normalize(inputs, axis=0),
            feature_idx=feature_idx,
            grid=grid,
        )
        ribbon_plot(
            x=grid,
            y=pdps,
            color=color,
            label=feature,
            samples_kwargs={"linewidth": 1},
        )

    plt.xlabel("Predictor value")  # pyright: ignore
    plt.ylabel("Marginal migration rate")  # pyright: ignore
    plt.yscale("log")  # pyright: ignore
    plt.legend()  # pyright: ignore
    plt.savefig(output_dir / "flights-and-populations-PDPs.svg")  # pyright: ignore
    plt.close()


def plot_eucovid():
    plot_eucovid_flights_over_population()
    plot_eucovid_flights_and_populations()

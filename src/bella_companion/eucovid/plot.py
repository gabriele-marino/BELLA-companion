import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from phylogenie import load_nexus
from phylogenie.draw import CalibrationNode, draw_colored_dated_tree_categorical

from bella_companion.backend import (
    MLPEnsemble,
    get_partial_dependence_plot_distribution,
    normalize,
    read_log_file,
    ribbon_plot,
)
from bella_companion.eucovid.settings import COLORS, COUNTRIES, DATA_DIR, N_COUNTRIES


def plot_eucovid_flights_over_populations():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    summaries_dir = (
        Path(os.environ["BELLA_SUMMARIES_DIR"]) / "eucovid" / "flights_over_populations"
    )

    log = read_log_file(summaries_dir / "GLM" / "MCMC.combined.log", burn_in=0.0)
    log = log.sample(n=100, random_state=42)  # pyright: ignore
    w = np.array(log["migrationRateW"])
    scaler = np.array(log["migrationRateScaler"])
    data_dir = DATA_DIR / "flights_over_populations"
    data = np.loadtxt(data_dir / "flights_over_populations.csv")
    x = np.linspace(np.min(data), np.max(data), 10)
    y = np.exp(np.log(scaler)[:, None] + np.outer(w, x))
    ribbon_plot(
        x=normalize(x), y=y, color="C1", label="GLM", samples_kwargs={"linewidth": 1}
    )

    mlps = MLPEnsemble.from_log_file(
        log_file=summaries_dir / "BELLA" / "MCMC.combined.log",
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

    summaries_dir = (
        Path(os.environ["BELLA_SUMMARIES_DIR"]) / "eucovid" / "flights_and_populations"
    )

    mlps = MLPEnsemble.from_log_file(
        log_file=summaries_dir / "BELLA" / "MCMC.combined.log",
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


def plot_eucovid_trees():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    summaries_dir = (
        Path(os.environ["BELLA_SUMMARIES_DIR"]) / "eucovid" / "flights_over_populations"
    )

    for model in ["GLM", "BELLA"]:
        tree = load_nexus(summaries_dir / model / "mcc.nexus")["TREE_MCC_CA"]
        tree.ladderize()

        # If a node has multiple equally probable countries, randomly assign one for visualization
        rng = np.random.default_rng(42)
        for node in tree:
            if "+" in node["type"]:
                countries = node["type"].split("+")
                node["type"] = rng.choice(countries)

        node1 = CalibrationNode(
            node=tree.get_node(name="CHN/WH-09/2020|China|2020-01-08"),
            date=date.fromisoformat("2020-01-08"),
        )
        node2 = CalibrationNode(
            node=tree.get_node(name="IMS-10216-CVDP-0161|Germany|2020-03-08"),
            date=date.fromisoformat("2020-03-08"),
        )

        plt.figure(figsize=(8, 12))  # pyright: ignore

        ax = draw_colored_dated_tree_categorical(
            tree=tree,
            calibration_nodes=(node1, node2),
            color_by="type",
            colormap=COLORS,
            legend_kwargs={"loc": "upper left"},
            branch_kwargs={"linewidth": 3},
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.savefig(output_dir / f"{model}-tree.svg")  # pyright: ignore


def plot_eucovid_sankey():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    runs_dir = (
        Path(os.environ["BELLA_BEAST_OUTPUT_DIR"])
        / "eucovid"
        / "flights_over_populations"
    )

    AGES = [
        0.173,  # 63 days
        0.115,  # 42 days
        0.058,  # 21 days
        0,
    ]
    N_LAYERS = len(AGES)
    N_TIME_BINS = N_LAYERS - 1

    for model in ["GLM", "BELLA"]:
        trajectories = pd.concat(
            [
                pd.read_csv(  # pyright: ignore
                    runs_dir / model / str(seed) / "trajectories.csv", sep="\t"
                ).assign(seed=seed)
                for seed in range(1, 4)
            ],
            ignore_index=True,
        )
        trajectories = trajectories[trajectories["Sample"] >= 1_000_000]
        trajectories_index = trajectories.set_index(["Sample", "seed"]).index.unique()

        def _count_migrations(source: str, target: str, time_bin: int) -> int:
            mask = (
                (trajectories["age"] < AGES[time_bin])
                & (trajectories["age"] > AGES[time_bin + 1])
                & (trajectories["variable"] == "M")
                & (trajectories["type"] == COUNTRIES.index(source))
                & (trajectories["type2"] == COUNTRIES.index(target))
            )
            migrations_per_traj = (
                trajectories.loc[mask].groupby(["Sample", "seed"])["value"].sum()  # pyright: ignore
            )
            n_migrations = int(
                migrations_per_traj.reindex(trajectories_index, fill_value=0).median()
            )
            return n_migrations

        migrations = np.array(
            [
                _count_migrations(source, target, time_bin)
                for time_bin in range(N_TIME_BINS)
                for source in COUNTRIES
                for target in COUNTRIES
                if target != source
            ]
        )
        mask = migrations > 0
        migrations = migrations[mask]
        source = np.repeat(list(range(N_COUNTRIES * N_TIME_BINS)), N_COUNTRIES - 1)[
            mask
        ]
        target = np.array(
            [
                x + N_COUNTRIES * t
                for t in range(1, N_TIME_BINS + 1)
                for i in list(range(N_COUNTRIES))
                for x in list(range(N_COUNTRIES))
                if x != i
            ]
        )[mask]

        nodes_mask = np.unique(np.concatenate((source, target)))
        colors = np.array(list(COLORS.values()) * N_LAYERS)
        x = np.repeat([0.1, 0.4, 0.7, 1], N_COUNTRIES)[nodes_mask]
        y = np.repeat([0.1, 0.2, 0.3, 0.4], N_COUNTRIES)[nodes_mask]

        fig = go.Figure(
            go.Sankey(  # pyright: ignore
                arrangement="snap",
                node=dict(color=colors, x=x, y=y),
                link=dict(source=source, target=target, value=migrations),
            )
        )
        fig.write_image(output_dir / f"{model}-sankey.svg")


def plot_eucovid():
    plot_eucovid_flights_over_populations()
    plot_eucovid_flights_and_populations()
    plot_eucovid_trees()
    plot_eucovid_sankey()

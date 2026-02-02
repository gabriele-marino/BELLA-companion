import os
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.patches import Patch
from phylogenie import load_nexus
from phylogenie.draw import CalibrationNode, draw_colored_dated_tree_categorical

from bella_companion.backend import (
    MLPEnsemble,
    get_partial_dependence_plot_distribution,
    normalize,
    read_log_file,
    ribbon_plot,
)
from bella_companion.eucovid.settings import (
    COLORS,
    COUNTRIES,
    DATA_DIR,
    N_COUNTRIES,
    N_SEEDS,
)


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
            node=tree.get_descendant(name="CHN/WH-09/2020|China|2020-01-08"),
            date=date.fromisoformat("2020-01-08"),
        )
        node2 = CalibrationNode(
            node=tree.get_descendant(name="IMS-10216-CVDP-0161|Germany|2020-03-08"),
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

    def _get_migrations(
        traj: pd.DataFrame, time_bin: int, source: str, target: str
    ) -> pd.Series:
        mask = (
            (traj["age"] < AGES[time_bin])
            & (traj["age"] > AGES[time_bin + 1])
            & (traj["variable"] == "M")
            & (traj["type"] == COUNTRIES.index(source))
            & (traj["type2"] == COUNTRIES.index(target))
        )
        migrations = traj.loc[mask].groupby(["Sample", "seed"])["value"].sum()  # pyright: ignore
        index = traj.set_index(["Sample", "seed"]).index.unique()
        return migrations.reindex(index, fill_value=0)

    trajectories = {
        model: pd.concat(
            [
                pd.read_csv(  # pyright: ignore
                    runs_dir / model / str(seed) / "trajectories.csv", sep="\t"
                ).assign(seed=seed)
                for seed in range(1, N_SEEDS + 1)
            ],
            ignore_index=True,
        ).query("Sample >= 1_000_000")
        for model in ["GLM", "BELLA"]
    }

    # Plot the estimated dates of introduction to Europe
    plt.figure(figsize=(8, 6))  # pyright: ignore
    ref_date = datetime(2020, 3, 8)  # date of the latest sample
    initial_introductions = {
        model: pd.to_datetime(
            [
                ref_date - timedelta(days=age * 365)
                for age in trajectories[model]
                .query(f"variable == 'M' and type == {COUNTRIES.index('China')}")  # pyright: ignore
                .groupby(["Sample", "seed"])["age"]
                .max()
                .values
            ]
        )
        for model in ["GLM", "BELLA"]
    }
    plt.hist(  # pyright: ignore
        [initial_introductions["GLM"], initial_introductions["BELLA"]],
        bins=20,
        label=["GLM", "BELLA"],
        color=["C1", "C2"],
        edgecolor="black",
        alpha=0.7,
    )
    plt.legend()  # pyright: ignore
    plt.xlabel("Estimated date of introduction to Europe")  # pyright: ignore
    plt.ylabel("Frequency")  # pyright: ignore
    plt.xticks(rotation=45)  # pyright: ignore
    plt.savefig(output_dir / "introduction-dates.svg")  # pyright: ignore
    plt.close()

    migrations = {
        model: {
            (time_bin, source, target): _get_migrations(
                traj=trajectories[model],
                time_bin=time_bin,
                source=source,
                target=target,
            )
            for time_bin in range(N_TIME_BINS)
            for source in COUNTRIES
            for target in COUNTRIES
            if source != target
        }
        for model in ["GLM", "BELLA"]
    }

    migrations_output_dir = output_dir / "migrations"
    os.makedirs(migrations_output_dir, exist_ok=True)

    # Plot migration distributions
    for time_bin in range(N_TIME_BINS):
        fig, axes = plt.subplots(  # pyright: ignore
            nrows=N_COUNTRIES, ncols=N_COUNTRIES - 1, figsize=(20, 15)
        )
        for i, source in enumerate(COUNTRIES):
            for j, target in enumerate([c for c in COUNTRIES if c != source]):
                ax = axes[i, j]
                max_value = pd.concat(
                    [
                        migrations[model][(time_bin, source, target)]
                        for model in ["GLM", "BELLA"]
                    ]
                ).max()
                for color, model in zip(["C1", "C2"], ["GLM", "BELLA"]):
                    n_migrations = migrations[model][(time_bin, source, target)]
                    n_migrations.plot(
                        kind="hist",
                        bins=np.linspace(0, max_value, 30),
                        density=True,
                        ax=ax,
                        alpha=0.6,
                        color=color,
                    )
                    try:
                        n_migrations.plot(kind="density", ax=ax, color=color)
                    except np.linalg.LinAlgError:
                        pass

                    ax.set_title(f"{source} → {target}")
                    ax.set_xlim(left=0)
                    ax.set_xlabel("N. migration events")

        legend_elements = [
            Patch(facecolor="C1", label="GLM", alpha=0.6),
            Patch(facecolor="C2", label="BELLA", alpha=0.6),
        ]
        fig.legend(  # pyright: ignore
            handles=legend_elements,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.05),
        )

        plt.savefig(migrations_output_dir / f"{time_bin}.svg")  # pyright: ignore
        plt.close(fig)

    # Plot sankey diagrams using median migrations
    for model in ["GLM", "BELLA"]:
        median_migrations = np.array(list(map(np.median, migrations[model].values())))
        mask = median_migrations > 0
        median_migrations = median_migrations[mask]
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
                link=dict(source=source, target=target, value=median_migrations),
            )
        )
        fig.write_image(output_dir / f"{model}-sankey.svg")


def plot_likelihood():
    output_dir = Path(os.environ["BELLA_FIGURES_DIR"]) / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    summaries_dir = (
        Path(os.environ["BELLA_SUMMARIES_DIR"]) / "eucovid" / "flights_over_populations"
    )

    glm_log = read_log_file(summaries_dir / "GLM" / "MCMC.combined.log", burn_in=0.0)
    bella_log = read_log_file(
        summaries_dir / "BELLA" / "MCMC.combined.log", burn_in=0.0
    )

    all_likelihoods = np.concatenate([glm_log["likelihood"], bella_log["likelihood"]])
    bins = np.histogram_bin_edges(all_likelihoods, bins=20).tolist()

    plt.figure(figsize=(10, 5))  # pyright: ignore
    plt.hist(  # pyright: ignore
        glm_log["likelihood"],
        bins=bins,
        color="C1",
        alpha=0.5,
        label="GLM",
        edgecolor="black",
        density=True,
    )
    plt.hist(  # pyright: ignore
        bella_log["likelihood"],
        bins=bins,
        color="C2",
        alpha=0.5,
        label="BELLA",
        edgecolor="black",
        density=True,
    )

    sns.kdeplot(glm_log["likelihood"].tolist(), color="C1", lw=2)
    sns.kdeplot(bella_log["likelihood"].tolist(), color="C2", lw=2)

    plt.xlabel("Phylogenetic Likelihood")  # pyright: ignore
    plt.ylabel("Density")  # pyright: ignore
    plt.legend()  # pyright: ignore
    plt.savefig(output_dir / "likelihood.svg")  # pyright: ignore


def plot_eucovid():
    plot_eucovid_flights_over_populations()
    plot_eucovid_flights_and_populations()
    plot_eucovid_trees()
    plot_eucovid_sankey()
    plot_likelihood()

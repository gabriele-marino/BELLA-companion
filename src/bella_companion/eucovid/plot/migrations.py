import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import Patch

from bella_companion.eucovid.settings import COLORS, COUNTRIES, N_COUNTRIES, N_SEEDS
from bella_companion.settings import settings
from bella_companion.typings import ModelID

TimeBin: TypeAlias = int
SourceCountry: TypeAlias = str
TargetCountry: TypeAlias = str
MigrationID: TypeAlias = tuple[TimeBin, SourceCountry, TargetCountry]
Migrations: TypeAlias = dict[MigrationID, pd.Series]

_AGES: list[float] = (np.array([63, 42, 21, 0]) / 365).tolist()
_N_LAYERS = len(_AGES)
_N_TIME_BINS = _N_LAYERS - 1


def _plot_introduction_dates(
    trajectories: dict[ModelID, pd.DataFrame], output_file: Path
):
    plt.figure(layout="constrained", figsize=(4, 3))  # pyright: ignore
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
    plt.savefig(output_file)  # pyright: ignore
    plt.close()


def _get_migrations(
    traj: pd.DataFrame, time_bin: int, source: str, target: str
) -> pd.Series:
    mask = (
        (traj["age"] < _AGES[time_bin])
        & (traj["age"] > _AGES[time_bin + 1])
        & (traj["variable"] == "M")
        & (traj["type"] == COUNTRIES.index(source))
        & (traj["type2"] == COUNTRIES.index(target))
    )
    migrations = traj.loc[mask].groupby(["Sample", "seed"])["value"].sum()  # pyright: ignore
    index = traj.set_index(["Sample", "seed"]).index.unique()
    return migrations.reindex(index, fill_value=0)


def _plot_migration_distributions(
    migrations: dict[ModelID, Migrations], output_dir: Path
):
    os.makedirs(output_dir, exist_ok=True)

    for time_bin in range(_N_TIME_BINS):
        fig, axes = plt.subplots(  # pyright: ignore
            nrows=N_COUNTRIES,
            ncols=N_COUNTRIES - 1,
            figsize=(20, 18),
            layout="constrained",
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

        plt.savefig(output_dir / f"time_bin_{time_bin}.pdf")  # pyright: ignore
        plt.close()


def _plot_eucovid_sankey(migrations: dict[ModelID, Migrations], output_file: Path):
    median_migrations = np.array(list(map(np.median, migrations["BELLA"].values())))
    mask = median_migrations > 0
    median_migrations = median_migrations[mask]
    source = np.repeat(list(range(N_COUNTRIES * _N_TIME_BINS)), N_COUNTRIES - 1)[mask]
    target = np.array(
        [
            x + N_COUNTRIES * t
            for t in range(1, _N_TIME_BINS + 1)
            for i in list(range(N_COUNTRIES))
            for x in list(range(N_COUNTRIES))
            if x != i
        ]
    )[mask]

    nodes_mask = np.unique(np.concatenate((source, target)))
    colors = np.array(list(COLORS.values()) * _N_LAYERS)
    x = np.repeat([0.1, 0.4, 0.7, 1], N_COUNTRIES)[nodes_mask]
    y = np.repeat([0.1, 0.2, 0.3, 0.4], N_COUNTRIES)[nodes_mask]

    fig = go.Figure(
        go.Sankey(  # pyright: ignore
            arrangement="snap",
            node=dict(color=colors, x=x, y=y),
            link=dict(source=source, target=target, value=median_migrations),
        )
    )
    fig.write_image(output_file)


def plot_migrations(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)

    runs_dir = settings.beast_output_dir / "eucovid"

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

    _plot_introduction_dates(trajectories, output_dir / "introduction_dates.pdf")

    migrations = {
        model: {
            (time_bin, source, target): _get_migrations(
                traj=trajectories[model],
                time_bin=time_bin,
                source=source,
                target=target,
            )
            for time_bin in range(_N_TIME_BINS)
            for source in COUNTRIES
            for target in COUNTRIES
            if source != target
        }
        for model in ["GLM", "BELLA"]
    }

    _plot_migration_distributions(migrations, output_dir / "distributions")

    _plot_eucovid_sankey(migrations, output_dir / "sankey.pdf")

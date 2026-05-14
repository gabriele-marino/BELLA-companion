import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from bella_companion.backend.plots import skyline_plot
from bella_companion.simulations.scenarios import (
    EPI_MULTITYPE,
    EPI_SKYLINE_SCENARIOS,
    EPI_TIME_AXIS,
    FBD_2TRAITS,
    FBD_NO_TRAITS_SCENARIOS,
    FBD_TIME_AXIS,
)
from bella_companion.targets import SkylineArray


def _plot_epi_skyline(ax: Axes, reproduction_number: SkylineArray):
    skyline_plot(
        reproduction_number, x=EPI_TIME_AXIS, step_kwargs={"color": "k"}, ax=ax
    )
    ax.set_ylabel(r"$R_t$")  # pyright: ignore
    ax.set_xlabel("Time")  # pyright: ignore


def _plot_epi_multitype(ax: Axes):
    sort_idx = np.argsort(EPI_MULTITYPE.migration_predictor.flatten())
    ax.plot(  # pyright: ignore
        EPI_MULTITYPE.migration_predictor.flatten()[sort_idx],
        EPI_MULTITYPE.migration_rates.flatten()[sort_idx],
        marker="o",
        color="k",
    )
    ax.set_xlabel(r"$x_{ij}$")  # pyright: ignore
    ax.set_ylabel(r"$m_{ij}$")  # pyright: ignore


def _plot_fbd_no_traits(ax: Axes, birth_rate: SkylineArray, death_rate: SkylineArray):
    skyline_plot(
        list(reversed(birth_rate)),
        x=FBD_TIME_AXIS,
        step_kwargs={"label": r"$\lambda$"},
        ax=ax,
    )
    skyline_plot(
        list(reversed(death_rate)),
        x=FBD_TIME_AXIS,
        step_kwargs={"label": r"$\mu$"},
        ax=ax,
    )
    ax.invert_xaxis()
    ax.set_ylabel("Rate")  # pyright: ignore
    ax.set_xlabel("Time")  # pyright: ignore
    ax.legend()  # pyright: ignore


def _plot_fbd_2traits(ax: Axes):
    skyline_plot(
        list(reversed(FBD_2TRAITS.birth_rate_trait1_unset)),
        x=FBD_TIME_AXIS,
        step_kwargs={"label": r"$\lambda^{(00)} = \lambda^{(01)}$", "color": "C0"},
        ax=ax,
    )
    skyline_plot(
        list(reversed(FBD_2TRAITS.birth_rate_trait1_set)),
        x=FBD_TIME_AXIS,
        step_kwargs={
            "label": r"$\lambda^{(10)} = \lambda^{(11)}$",
            "color": "C0",
            "linestyle": "dashed",
        },
        ax=ax,
    )
    skyline_plot(
        list(reversed(FBD_2TRAITS.death_rate_trait1_unset)),
        x=FBD_TIME_AXIS,
        step_kwargs={"label": r"$\mu^{(00)} = \mu^{(01)}$", "color": "C1"},
        ax=ax,
    )
    skyline_plot(
        list(reversed(FBD_2TRAITS.death_rate_trait1_set)),
        x=FBD_TIME_AXIS,
        step_kwargs={
            "label": r"$\mu^{(10)} = \mu^{(11)}$",
            "color": "C1",
            "linestyle": "dashed",
        },
        ax=ax,
    )

    ax.invert_xaxis()
    ax.set_ylabel("Rate")  # pyright: ignore
    ax.set_xlabel("Time")  # pyright: ignore
    ax.legend()  # pyright: ignore


def plot_scenarios(output_file: Path):
    _, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")  # pyright: ignore

    for ax, label in zip(axes.flat, string.ascii_lowercase):
        ax.text(
            -0.22, 0.95, label, transform=ax.transAxes, fontsize=15, fontweight="bold"
        )

    for i, scenario in enumerate(EPI_SKYLINE_SCENARIOS.values()):
        _plot_epi_skyline(axes[0, i], scenario.reproduction_number)
    _plot_epi_multitype(axes[0, 3])

    for i, scenario in enumerate(FBD_NO_TRAITS_SCENARIOS.values()):
        _plot_fbd_no_traits(axes[1, i], scenario.birth_rate, scenario.death_rate)
    _plot_fbd_2traits(axes[1, 3])

    plt.savefig(output_file)  # pyright: ignore
    plt.close()

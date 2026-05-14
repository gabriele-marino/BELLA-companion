import os
from argparse import ArgumentParser

from bella_companion.settings import settings
from bella_companion.simulations.plot.epi_skyline.metrics import (
    plot_metrics_through_time,
)
from bella_companion.simulations.plot.epi_skyline.ribbons import plot_ribbons
from bella_companion.simulations.plot.epi_skyline.sensitivity import (
    plot_sensitivity_ribbons,
)


def register_epi_skyline_plot_cli(epi_skyline_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "epi-skyline"
    os.makedirs(output_dir, exist_ok=True)

    epi_skyline_plot_subparsers = epi_skyline_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    epi_skyline_plot_subparsers.add_parser(
        "ribbons", help="Generate ribbon plots for the epi-skyline scenarios."
    ).set_defaults(func=lambda: plot_ribbons(output_dir / "ribbons.pdf"))

    epi_skyline_plot_subparsers.add_parser(
        "metrics", help="Generate metrics plots for the epi-skyline scenarios."
    ).set_defaults(func=lambda: plot_metrics_through_time(output_dir / "metrics.pdf"))

    epi_skyline_plot_subparsers.add_parser(
        "sensitivity",
        help="Generate sensitivity ribbon plots for the epi-skyline scenarios.",
    ).set_defaults(func=lambda: plot_sensitivity_ribbons(output_dir / "sensitivity"))

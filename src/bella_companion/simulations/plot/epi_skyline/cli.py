from argparse import ArgumentParser

from bella_companion.simulations.plot.epi_skyline.metrics import (
    plot_metrics_through_time,
)
from bella_companion.simulations.plot.epi_skyline.ribbons import plot_ribbons


def register_epi_skyline_plot_cli(epi_skyline_plot_parser: ArgumentParser):
    epi_skyline_plot_subparsers = epi_skyline_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    epi_skyline_plot_subparsers.add_parser(
        "ribbons", help="Generate ribbon plots for the epi-skyline scenarios."
    ).set_defaults(func=plot_ribbons)

    epi_skyline_plot_subparsers.add_parser(
        "metrics", help="Generate metrics plots for the epi-skyline scenarios."
    ).set_defaults(func=plot_metrics_through_time)

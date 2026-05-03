from argparse import ArgumentParser

from bella_companion.simulations.plot.fbd_no_traits.metrics import (
    plot_metrics_through_time,
)
from bella_companion.simulations.plot.fbd_no_traits.ribbons import plot_ribbons


def register_fbd_no_traits_plot_cli(fbd_no_traits_plot_parser: ArgumentParser):
    fbd_no_traits_plot_subparsers = fbd_no_traits_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    fbd_no_traits_plot_subparsers.add_parser(
        "ribbons", help="Generate ribbon plots for the fbd-no-traits scenarios."
    ).set_defaults(func=plot_ribbons)

    fbd_no_traits_plot_subparsers.add_parser(
        "metrics", help="Generate metrics plots for the fbd-no-traits scenarios."
    ).set_defaults(func=plot_metrics_through_time)

import os
from argparse import ArgumentParser

from bella_companion.settings import settings
from bella_companion.simulations.plot.fbd_no_traits.metrics import (
    plot_metrics_through_time,
)
from bella_companion.simulations.plot.fbd_no_traits.ribbons import plot_ribbons
from bella_companion.simulations.plot.fbd_no_traits.sensitivity import (
    plot_sensitivity_ribbons,
)


def register_fbd_no_traits_plot_cli(fbd_no_traits_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "fbd-no-traits"
    os.makedirs(output_dir, exist_ok=True)

    fbd_no_traits_plot_subparsers = fbd_no_traits_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    fbd_no_traits_plot_subparsers.add_parser(
        "ribbons", help="Generate ribbon plots for the fbd-no-traits scenarios."
    ).set_defaults(func=lambda: plot_ribbons(output_dir / "ribbons.pdf"))

    fbd_no_traits_plot_subparsers.add_parser(
        "metrics", help="Generate metrics plots for the fbd-no-traits scenarios."
    ).set_defaults(func=lambda: plot_metrics_through_time(output_dir / "metrics.pdf"))

    fbd_no_traits_plot_subparsers.add_parser(
        "sensitivity",
        help="Generate sensitivity ribbon plots for the fbd-no-traits scenarios.",
    ).set_defaults(func=lambda: plot_sensitivity_ribbons(output_dir / "sensitivity"))

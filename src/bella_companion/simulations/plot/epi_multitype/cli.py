import os
from argparse import ArgumentParser

from bella_companion.settings import settings
from bella_companion.simulations.plot.epi_multitype.sensitivity import (
    plot_estimate_sensitivity,
)
from bella_companion.simulations.plot.epi_multitype.summary import plot_summary


def register_epi_multitype_plot_cli(epi_multitype_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "epi-multitype"
    os.makedirs(output_dir, exist_ok=True)

    epi_multitype_plot_subparsers = epi_multitype_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    epi_multitype_plot_subparsers.add_parser(
        "summary", help="Generate summary plots for the epi-multitype scenario."
    ).set_defaults(func=lambda: plot_summary(output_dir / "summary.pdf"))

    epi_multitype_plot_subparsers.add_parser(
        "sensitivity", help="Generate sensitivity plots for the epi-multitype scenario."
    ).set_defaults(
        func=lambda: plot_estimate_sensitivity(output_dir / "sensitivity.pdf")
    )

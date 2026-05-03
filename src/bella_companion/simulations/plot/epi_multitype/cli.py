from argparse import ArgumentParser

from bella_companion.simulations.plot.epi_multitype.summary import plot_summary


def register_epi_multitype_plot_cli(epi_multitype_plot_parser: ArgumentParser):
    epi_multitype_plot_subparsers = epi_multitype_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    epi_multitype_plot_subparsers.add_parser(
        "summary", help="Generate summary plots for the epi-multitype scenario."
    ).set_defaults(func=plot_summary)

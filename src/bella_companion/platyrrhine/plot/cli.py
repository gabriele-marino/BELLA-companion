from argparse import ArgumentParser

from bella_companion.platyrrhine.plot.summary import plot_summary


def register_plot_cli(platyrrhine_plot_parser: ArgumentParser):
    platyrrhine_plot_subparsers = platyrrhine_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    platyrrhine_plot_subparsers.add_parser(
        "summary", help="Generate summary plots for the platyrrhine analyses."
    ).set_defaults(func=plot_summary)

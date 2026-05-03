from argparse import ArgumentParser

from bella_companion.platyrrhine.plot.cli import register_plot_cli
from bella_companion.platyrrhine.run import run
from bella_companion.platyrrhine.summarize import summarize


def register_platyrrhine_cli(platyrrhine_parser: ArgumentParser):
    platyrrhine_subparser = platyrrhine_parser.add_subparsers(
        dest="subcommand", required=True
    )

    platyrrhine_subparser.add_parser(
        "run", help="Run BEAST2 analyses on empirical platyrrhine datasets."
    ).set_defaults(func=run)

    platyrrhine_subparser.add_parser(
        "summarize",
        help="Summarize BEAST2 log outputs for empirical platyrrhine datasets.",
    ).set_defaults(func=summarize)

    platyrrhine_plot_parser = platyrrhine_subparser.add_parser(
        "plot", help="Generate plots and figures for empirical platyrrhine datasets."
    )
    register_plot_cli(platyrrhine_plot_parser)

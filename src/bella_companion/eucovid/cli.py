from argparse import ArgumentParser

from bella_companion.eucovid.plot.cli import register_plot_cli
from bella_companion.eucovid.run import run
from bella_companion.eucovid.summarize import summarize


def register_eucovid_cli(eucovid_parser: ArgumentParser):
    eucovid_subparsers = eucovid_parser.add_subparsers(dest="subcommand", required=True)

    eucovid_subparsers.add_parser(
        "run", help="Run BEAST2 analyses on empirical eucovid datasets."
    ).set_defaults(func=run)

    eucovid_subparsers.add_parser(
        "summarize", help="Summarize BEAST2 analyses on empirical eucovid datasets."
    ).set_defaults(func=summarize)

    eucovid_plot_parser = eucovid_subparsers.add_parser(
        "plot", help="Generate plots and figures for empirical eucovid datasets."
    )
    register_plot_cli(eucovid_plot_parser)

from argparse import ArgumentParser

from bella_companion.eucovid.plot import (
    plot_eucovid,
    plot_eucovid_flights_and_populations,
    plot_eucovid_flights_over_populations,
    plot_eucovid_sankey,
    plot_eucovid_trees,
    plot_likelihood,
)
from bella_companion.eucovid.run import run_eucovid
from bella_companion.eucovid.summarize import summarize_eucovid


def register_eucovid_cli(eucovid_parser: ArgumentParser):
    eucovid_subparsers = eucovid_parser.add_subparsers(dest="subcommand", required=True)

    eucovid_subparsers.add_parser(
        "run", help="Run BEAST2 analyses on empirical eucovid datasets."
    ).set_defaults(func=run_eucovid)

    eucovid_subparsers.add_parser(
        "summarize",
        help="Summarize BEAST2 log outputs for empirical eucovid datasets.",
    ).set_defaults(func=summarize_eucovid)

    eucovid_plot_parser = eucovid_subparsers.add_parser(
        "plot", help="Generate plots and figures for empirical eucovid datasets."
    )
    eucovid_plot_subparsers = eucovid_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    eucovid_plot_subparsers.add_parser(
        "all", help="Generate plots and figures for empirical eucovid datasets."
    ).set_defaults(func=plot_eucovid)

    eucovid_plot_subparsers.add_parser(
        "likelihood", help="Generate likelihood distribution plots for eucovid dataset."
    ).set_defaults(func=plot_likelihood)

    eucovid_plot_subparsers.add_parser(
        "sankey", help="Generate sankey plots for eucovid dataset."
    ).set_defaults(func=plot_eucovid_sankey)

    eucovid_plot_subparsers.add_parser(
        "trees", help="Generate tree plots for eucovid dataset."
    ).set_defaults(func=plot_eucovid_trees)

    eucovid_plot_subparsers.add_parser(
        "flights-and-populations",
        help="Generate plots for eucovid dataset in the flights and populations scenario.",
    ).set_defaults(func=plot_eucovid_flights_and_populations)

    eucovid_plot_subparsers.add_parser(
        "flights-over-populations",
        help="Generate plots for eucovid dataset in the flights over populations scenario.",
    ).set_defaults(func=plot_eucovid_flights_over_populations)

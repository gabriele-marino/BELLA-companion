from argparse import ArgumentParser

from bella_companion.simulations.generate import generate
from bella_companion.simulations.plot import register_plot_cli
from bella_companion.simulations.run import run
from bella_companion.simulations.summarize import summarize
from bella_companion.simulations.tables import build_tables


def register_sim_cli(sim_parser: ArgumentParser):
    sim_subparsers = sim_parser.add_subparsers(dest="subcommand", required=True)

    sim_subparsers.add_parser(
        "generate", help="Generate synthetic simulation datasets."
    ).set_defaults(func=generate)

    sim_subparsers.add_parser(
        "run", help="Run BEAST2 analyses on simulated datasets."
    ).set_defaults(func=run)

    sim_subparsers.add_parser(
        "summarize", help="Summarize BEAST2 log outputs for simulated datasets."
    ).set_defaults(func=summarize)

    sim_plot_parser = sim_subparsers.add_parser(
        "plot", help="Generate plots and figures for simulated datasets."
    )
    register_plot_cli(sim_plot_parser)

    sim_subparsers.add_parser(
        "tables",
        help="Build metrics tables summarizing the results for simulated datasets.",
    ).set_defaults(func=build_tables)

import argparse

from bella_companion.eucovid import register_eucovid_cli
from bella_companion.platyrrhine import register_platyrrhine_cli
from bella_companion.simulations import register_sim_cli


def main():
    parser = argparse.ArgumentParser(
        prog="bella",
        description="Companion tool with experiments and evaluation for Bayesian Evolutionary Layered Learning Architectures (BELLA) BEAST2 package.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("sim", help="Simulation workflows.")
    register_sim_cli(sim_parser)

    platyrrhine_parser = subparsers.add_parser(
        "platyrrhine", help="Empirical platyrrhine workflows."
    )
    register_platyrrhine_cli(platyrrhine_parser)

    eucovid_parser = subparsers.add_parser(
        "eucovid", help="Empirical eucovid workflows."
    )
    register_eucovid_cli(eucovid_parser)

    args = parser.parse_args()
    args.func()

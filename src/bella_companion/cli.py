import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from bella_companion.eucovid import (
    plot_eucovid,
    plot_eucovid_flights_and_populations,
    plot_eucovid_flights_over_populations,
    plot_eucovid_sankey,
    plot_eucovid_trees,
    plot_likelihood,
    run_eucovid,
    summarize_eucovid,
)
from bella_companion.platyrrhine import (
    plot_platyrrhine,
    plot_platyrrhine_estimates,
    plot_platyrrhine_shap,
    plot_platyrrhine_trees,
    run_platyrrhine,
    summarize_platyrrhine,
)
from bella_companion.simulations import (
    generate,
    plot_epi_multitype,
    plot_epi_skyline,
    plot_fbd_2traits,
    plot_fbd_no_traits,
    plot_scenarios,
    plot_simulations,
    run_metrics,
    run_simulations,
    summarize_simulations,
)


def main():
    load_dotenv(Path(os.getcwd()) / ".env")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["font.size"] = 20
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["lines.linewidth"] = 4

    parser = argparse.ArgumentParser(
        prog="bella",
        description="Companion tool with experiments and evaluation for Bayesian Evolutionary Layered Learning Architectures (BELLA) BEAST2 package.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------
    # Simulation datasets
    # -------------------

    sim_parser = subparsers.add_parser("sim", help="Simulation workflows")
    sim_subparsers = sim_parser.add_subparsers(dest="subcommand", required=True)

    sim_subparsers.add_parser(
        "generate", help="Generate synthetic simulation datasets."
    ).set_defaults(func=generate)

    sim_subparsers.add_parser(
        "run", help="Run BEAST2 analyses on simulated datasets."
    ).set_defaults(func=run_simulations)

    sim_subparsers.add_parser(
        "summarize", help="Summarize BEAST2 log outputs for simulated datasets."
    ).set_defaults(func=summarize_simulations)

    sim_subparsers.add_parser(
        "metrics", help="Compute and print metrics for simulated datasets."
    ).set_defaults(func=run_metrics)

    sim_plot_parser = sim_subparsers.add_parser(
        "plot", help="Generate plots and figures for simulated datasets."
    )
    sim_plot_subparsers = sim_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    sim_plot_subparsers.add_parser(
        "all", help="Generate plots and figures for all simulation scenarios."
    ).set_defaults(func=plot_simulations)

    sim_plot_subparsers.add_parser(
        "epi-multitype", help="Generate plots for the epi-multitype scenario."
    ).set_defaults(func=plot_epi_multitype)

    sim_plot_subparsers.add_parser(
        "epi-skyline", help="Generate plots for the epi-skyline scenarios."
    ).set_defaults(func=plot_epi_skyline)

    sim_plot_subparsers.add_parser(
        "fbd-2traits", help="Generate plots for the fbd-2traits scenario."
    ).set_defaults(func=plot_fbd_2traits)

    sim_plot_subparsers.add_parser(
        "fbd-no-traits", help="Generate plots for the fbd-no-traits scenarios."
    ).set_defaults(func=plot_fbd_no_traits)

    sim_plot_subparsers.add_parser(
        "scenarios", help="Generate scenario overview plots."
    ).set_defaults(func=plot_scenarios)

    # -------------------
    # Platyrrhine dataset
    # -------------------

    platyrrhine_parser = subparsers.add_parser(
        "platyrrhine", help="Empirical platyrrhine datasets workflows"
    )
    platyrrhine_subparser = platyrrhine_parser.add_subparsers(
        dest="subcommand", required=True
    )

    platyrrhine_subparser.add_parser(
        "run", help="Run BEAST2 analyses on empirical platyrrhine datasets."
    ).set_defaults(func=run_platyrrhine)

    platyrrhine_subparser.add_parser(
        "summarize",
        help="Summarize BEAST2 log outputs for empirical platyrrhine datasets.",
    ).set_defaults(func=summarize_platyrrhine)

    platyrrhine_plot_parser = platyrrhine_subparser.add_parser(
        "plot", help="Generate plots and figures for empirical platyrrhine datasets."
    )
    platyrrhine_plot_subparsers = platyrrhine_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    platyrrhine_plot_subparsers.add_parser(
        "all", help="Generate plots and figures for empirical platyrrhine datasets."
    ).set_defaults(func=plot_platyrrhine)

    platyrrhine_plot_subparsers.add_parser(
        "estimates",
        help="Generate plots for parameter estimates for empirical platyrrhine datasets.",
    ).set_defaults(func=plot_platyrrhine_estimates)

    platyrrhine_plot_subparsers.add_parser(
        "trees",
        help="Generate plots for tree-mapped parameter estimates for empirical platyrrhine datasets.",
    ).set_defaults(func=plot_platyrrhine_trees)

    platyrrhine_plot_subparsers.add_parser(
        "shap",
        help="Generate SHAP plots for empirical platyrrhine datasets.",
    ).set_defaults(func=plot_platyrrhine_shap)

    # ---------------
    # EUCOVID dataset
    # ---------------

    eucovid_parser = subparsers.add_parser(
        "eucovid", help="Empirical eucovid workflows"
    )
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

    args = parser.parse_args()
    args.func()

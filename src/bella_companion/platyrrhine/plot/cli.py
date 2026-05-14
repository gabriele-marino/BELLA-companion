import os
from argparse import ArgumentParser

from bella_companion.platyrrhine.plot.marginal import plot_marginal_rates
from bella_companion.platyrrhine.plot.ribbons import plot_ribbons
from bella_companion.platyrrhine.plot.summary import plot_summary
from bella_companion.platyrrhine.plot.trees import plot_trees
from bella_companion.settings import settings


def register_plot_cli(platyrrhine_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "platyrrhine"
    os.makedirs(output_dir, exist_ok=True)

    platyrrhine_plot_subparsers = platyrrhine_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    platyrrhine_plot_subparsers.add_parser(
        "summary", help="Generate summary plots for the platyrrhine analyses."
    ).set_defaults(func=lambda: plot_summary(output_dir / "summary.pdf"))

    platyrrhine_plot_subparsers.add_parser(
        "ribbons", help="Generate ribbon plots for the platyrrhine analyses."
    ).set_defaults(func=lambda: plot_ribbons(output_dir / "ribbons.pdf"))

    platyrrhine_plot_subparsers.add_parser(
        "marginal",
        help="Generate marginal rates plots for the platyrrhine analyses.",
    ).set_defaults(func=lambda: plot_marginal_rates(output_dir / "marginal.pdf"))

    platyrrhine_plot_subparsers.add_parser(
        "trees", help="Generate colored trees for the platyrrhine analyses."
    ).set_defaults(func=lambda: plot_trees(output_dir / "trees.pdf"))

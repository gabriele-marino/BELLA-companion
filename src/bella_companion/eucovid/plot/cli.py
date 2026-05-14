import os
from argparse import ArgumentParser

from bella_companion.eucovid.plot.likelihood import plot_likelihood
from bella_companion.eucovid.plot.migrations import plot_migrations
from bella_companion.eucovid.plot.pdp import plot_pdp
from bella_companion.eucovid.plot.trees import plot_trees
from bella_companion.settings import settings


def register_plot_cli(eucovid_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "eucovid"
    os.makedirs(output_dir, exist_ok=True)

    eucovid_plot_subparsers = eucovid_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    eucovid_plot_subparsers.add_parser(
        "likelihood", help="Generate likelihood plots for the eucovid analyses."
    ).set_defaults(func=lambda: plot_likelihood(output_dir / "likelihood.pdf"))

    eucovid_plot_subparsers.add_parser(
        "trees", help="Generate tree plots for the eucovid analyses."
    ).set_defaults(func=lambda: plot_trees(output_dir / "trees"))

    eucovid_plot_subparsers.add_parser(
        "pdp", help="Generate partial dependence plots for the eucovid analyses."
    ).set_defaults(func=lambda: plot_pdp(output_dir / "PDP.pdf"))

    eucovid_plot_subparsers.add_parser(
        "migrations", help="Generate migration plots for the eucovid analyses."
    ).set_defaults(func=lambda: plot_migrations(output_dir / "migrations"))

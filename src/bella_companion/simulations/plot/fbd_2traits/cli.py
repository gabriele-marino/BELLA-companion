from argparse import ArgumentParser

from bella_companion.simulations.plot.fbd_2traits.summary import plot_summary


def register_fbd_2traits_plot_cli(fbd_2traits_plot_parser: ArgumentParser):
    fbd_2traits_plot_subparsers = fbd_2traits_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    fbd_2traits_plot_subparsers.add_parser(
        "summary", help="Generate ribbon plots for the fbd-2traits scenarios."
    ).set_defaults(func=plot_summary)

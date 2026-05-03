from argparse import ArgumentParser

from bella_companion.simulations.plot.epi_multitype import (
    register_epi_multitype_plot_cli,
)
from bella_companion.simulations.plot.epi_skyline import register_epi_skyline_plot_cli
from bella_companion.simulations.plot.fbd_2traits import register_fbd_2traits_plot_cli
from bella_companion.simulations.plot.fbd_no_traits import (
    register_fbd_no_traits_plot_cli,
)
from bella_companion.simulations.plot.metrics import plot_metrics
from bella_companion.simulations.plot.scenarios import plot_scenarios


def register_plot_cli(sim_plot_parser: ArgumentParser):
    sim_plot_subparsers = sim_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    sim_plot_subparsers.add_parser(
        "scenarios", help="Generate scenario overview plots."
    ).set_defaults(func=plot_scenarios)

    sim_plot_subparsers.add_parser(
        "metrics",
        help="Generate plots summarizing the metrics of the models across all simulation scenarios.",
    ).set_defaults(func=plot_metrics)

    epi_skyline_plot_parser = sim_plot_subparsers.add_parser(
        "epi-skyline", help="Generate plots for the epi-skyline scenarios."
    )
    register_epi_skyline_plot_cli(epi_skyline_plot_parser)

    epi_multitype_plot_parser = sim_plot_subparsers.add_parser(
        "epi-multitype", help="Generate plots for the epi-multitype scenarios."
    )
    register_epi_multitype_plot_cli(epi_multitype_plot_parser)

    fbd_no_traits_plot_parser = sim_plot_subparsers.add_parser(
        "fbd-no-traits", help="Generate plots for the fbd-no-traits scenarios."
    )
    register_fbd_no_traits_plot_cli(fbd_no_traits_plot_parser)

    fbd_2traits_plot_parser = sim_plot_subparsers.add_parser(
        "fbd-2traits", help="Generate plots for the fbd-2traits scenarios."
    )
    register_fbd_2traits_plot_cli(fbd_2traits_plot_parser)

import os
from argparse import ArgumentParser

from bella_companion.settings import settings
from bella_companion.simulations.plot.fbd_2traits.benchmark import (
    plot_benchmark_ribbons,
)
from bella_companion.simulations.plot.fbd_2traits.metrics import (
    plot_metrics_through_time,
)
from bella_companion.simulations.plot.fbd_2traits.sensitivity import (
    plot_sensitivity_ribbons,
)
from bella_companion.simulations.plot.fbd_2traits.summary import plot_summary


def register_fbd_2traits_plot_cli(fbd_2traits_plot_parser: ArgumentParser):
    output_dir = settings.figures_dir / "fbd-2traits"
    os.makedirs(output_dir, exist_ok=True)

    fbd_2traits_plot_subparsers = fbd_2traits_plot_parser.add_subparsers(
        dest="subcommand", required=True
    )

    fbd_2traits_plot_subparsers.add_parser(
        "summary", help="Generate ribbon plots for the fbd-2traits scenarios."
    ).set_defaults(func=lambda: plot_summary(output_dir / "summary.pdf"))

    fbd_2traits_plot_subparsers.add_parser(
        "metrics", help="Generate metrics plots for the fbd-2traits scenarios."
    ).set_defaults(func=lambda: plot_metrics_through_time(output_dir / "metrics.pdf"))

    fbd_2traits_plot_subparsers.add_parser(
        "sensitivity",
        help="Generate sensitivity ribbon plots for the fbd-2traits scenarios.",
    ).set_defaults(
        func=lambda: plot_sensitivity_ribbons(output_dir / "sensitivity.pdf")
    )

    fbd_2traits_plot_subparsers.add_parser(
        "benchmark",
        help="Generate PA and GLM benchmark ribbon plots for the fbd-2traits scenarios.",
    ).set_defaults(func=lambda: plot_benchmark_ribbons(output_dir / "benchmark.pdf"))

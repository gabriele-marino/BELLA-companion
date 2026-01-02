from bella_companion.simulations.plot.epi_multitype import plot_epi_multitype
from bella_companion.simulations.plot.epi_skyline import plot_epi_skyline
from bella_companion.simulations.plot.fbd_2traits import plot_fbd_2traits
from bella_companion.simulations.plot.fbd_no_traits import plot_fbd_no_traits
from bella_companion.simulations.plot.scenarios import plot_scenarios


def plot_simulations():
    plot_epi_multitype()
    plot_epi_skyline()
    plot_fbd_2traits()
    plot_fbd_no_traits()
    plot_scenarios()


__all__ = [
    "plot_simulations",
    "plot_epi_multitype",
    "plot_epi_skyline",
    "plot_fbd_2traits",
    "plot_fbd_no_traits",
    "plot_scenarios",
]

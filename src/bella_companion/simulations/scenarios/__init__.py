from bella_companion.simulations.scenarios.common import (
    EPI_TIME_AXIS,
    FBD_RATE_UPPER,
    FBD_TIME_AXIS,
    MIGRATION_RATE_UPPER,
    N_TIME_BINS,
    REPRODUCTION_NUMBER_UPPER,
)
from bella_companion.simulations.scenarios.epi_multitype import EPI_MULTITYPE
from bella_companion.simulations.scenarios.epi_skyline import EPI_SKYLINE_SCENARIOS
from bella_companion.simulations.scenarios.fbd_2traits import FBD_2TRAITS
from bella_companion.simulations.scenarios.fbd_no_traits import FBD_NO_TRAITS_SCENARIOS
from bella_companion.typings import Scenario, ScenarioID

SCENARIOS: dict[ScenarioID, Scenario] = {
    **EPI_SKYLINE_SCENARIOS,
    "epi-multitype": EPI_MULTITYPE,
    **FBD_NO_TRAITS_SCENARIOS,
    "fbd-2traits": FBD_2TRAITS,
}
__all__ = [
    "EPI_TIME_AXIS",
    "FBD_RATE_UPPER",
    "FBD_TIME_AXIS",
    "MIGRATION_RATE_UPPER",
    "N_TIME_BINS",
    "REPRODUCTION_NUMBER_UPPER",
    "SCENARIOS",
    "EPI_MULTITYPE",
    "EPI_SKYLINE_SCENARIOS",
    "FBD_NO_TRAITS_SCENARIOS",
    "FBD_2TRAITS",
]

from dataclasses import dataclass
from functools import partial

import numpy as np
from phylogenie.skyline import SkylineParameter
from phylogenie.treesimulator.open_population import (
    get_bd_model,
)

from bella_companion.simulations.scenarios.common import (
    BECOME_UNINFECTIOUS_RATE,
    EPI_CHANGE_TIMES,
    EPI_MAX_TIME,
    EPI_SAMPLING_PROPORTION,
    N_TIME_BINS,
    REPRODUCTION_NUMBER_UPPER,
    get_last_sample_time,
    get_prior_params,
    get_random_time_series_predictor,
)
from bella_companion.targets import SkylineTarget
from bella_companion.typings import Scenario, SkylineArray


@dataclass(kw_only=True)
class EpiSkyline(Scenario):
    reproduction_number: SkylineArray

    @property
    def targets(self) -> list[SkylineTarget]:
        return [
            SkylineTarget(id="reproductionNumberSP", skyline=self.reproduction_number)
        ]


def _get_scenario(reproduction_number: SkylineArray) -> EpiSkyline:
    return EpiSkyline(
        name="epi-skyline",
        model=get_bd_model(
            reproduction_number=SkylineParameter(
                reproduction_number.tolist(), EPI_CHANGE_TIMES
            ),
            infectious_period=1 / BECOME_UNINFECTIOUS_RATE,
            sampling_proportion=EPI_SAMPLING_PROPORTION,
        ),
        max_time=EPI_MAX_TIME,
        reproduction_number=reproduction_number,
        beast_static_data={
            "processLength": EPI_MAX_TIME,
            "changeTimes": " ".join(map(str, EPI_CHANGE_TIMES)),
            **get_prior_params(
                "reproductionNumber", REPRODUCTION_NUMBER_UPPER, N_TIME_BINS
            ),
            "becomeUninfectiousRate": BECOME_UNINFECTIOUS_RATE,
            "samplingProportion": EPI_SAMPLING_PROPORTION,
            "timePredictor": " ".join(map(str, np.linspace(0, 1, N_TIME_BINS))),
        },
        beast_tree_data={"lastSampleTime": get_last_sample_time},
        beast_sample_data={
            "randomPredictor": partial(
                get_random_time_series_predictor, n_time_bins=N_TIME_BINS
            )
        },
    )


_REPRODUCTION_NUMBERS: list[SkylineArray] = [
    np.array([1.2] * 10),
    np.linspace(1.5, 1.0, 10, dtype=np.float64),
    np.concatenate(
        (np.linspace(1.2, 1.5, 5), np.linspace(1.5, 1.0, 5)), dtype=np.float64
    ),
]
EPI_SKYLINE_SCENARIOS = {
    f"epi-skyline_{i}": _get_scenario(r)
    for i, r in enumerate(_REPRODUCTION_NUMBERS, start=1)
}

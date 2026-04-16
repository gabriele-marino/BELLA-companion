from functools import partial

import numpy as np
from phylogenie.skyline import SkylineParameter
from phylogenie.treesimulator.open_population import (
    get_bd_model,
)

from bella_companion.simulations.scenarios.globals import (
    BECOME_UNINFECTIOUS_RATE,
    EPI_MAX_TIME,
    EPI_SAMPLING_PROPORTION,
)
from bella_companion.simulations.scenarios.scenario import Scenario
from bella_companion.simulations.scenarios.utils import (
    get_last_sample_time,
    get_prior_params,
    get_random_time_series_predictor,
)


def _get_scenario(reproduction_number: list[float]) -> Scenario:
    n_time_bins = len(reproduction_number)
    change_times = np.linspace(0, EPI_MAX_TIME, n_time_bins + 1)[1:-1].tolist()
    return Scenario(
        model=get_bd_model(
            reproduction_number=SkylineParameter(reproduction_number, change_times),
            infectious_period=1 / BECOME_UNINFECTIOUS_RATE,
            sampling_proportion=EPI_SAMPLING_PROPORTION,
        ),
        max_time=EPI_MAX_TIME,
        targets={
            "reproductionNumber": {
                f"reproductionNumberSPi{i}": r
                for i, r in enumerate(reproduction_number)
            }
        },
        beast_configs="epi-skyline",
        beast_args={
            "processLength": EPI_MAX_TIME,
            "changeTimes": " ".join(map(str, change_times)),
            **get_prior_params(
                "reproductionNumber", REPRODUCTION_NUMBER_UPPER, n_time_bins
            ),
            "becomeUninfectiousRate": BECOME_UNINFECTIOUS_RATE,
            "samplingProportion": EPI_SAMPLING_PROPORTION,
            "timePredictor": " ".join(map(str, np.linspace(0, 1, n_time_bins))),
        },
        tree_beast_args={"lastSampleTime": get_last_sample_time},
        get_random_predictor=partial(
            get_random_time_series_predictor, n_time_bins=n_time_bins
        ),
    )


REPRODUCTION_NUMBERS: list[list[float]] = [
    [1.2] * 10,
    np.linspace(1.5, 1.0, 10).tolist(),
    np.linspace(1.2, 1.5, 5).tolist() + np.linspace(1.5, 1.0, 5).tolist(),
]
REPRODUCTION_NUMBER_UPPER = 5
EPI_SKYLINE_SCENARIOS = [_get_scenario(r) for r in REPRODUCTION_NUMBERS]

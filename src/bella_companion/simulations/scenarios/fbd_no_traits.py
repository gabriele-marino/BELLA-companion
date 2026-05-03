from dataclasses import dataclass
from functools import partial

import numpy as np
from phylogenie.skyline import SkylineParameter
from phylogenie.treesimulator import Sampling, TimedEvent
from phylogenie.treesimulator.open_population import get_canonical_model

from bella_companion.simulations.scenarios.common import (
    FBD_CHANGE_TIMES,
    FBD_MAX_TIME,
    FBD_RATE_UPPER,
    FBD_SAMPLING_RATE,
    N_TIME_BINS,
    get_prior_params,
    get_random_time_series_predictor,
)
from bella_companion.targets import SkylineTarget
from bella_companion.typings import Scenario, SkylineArray


@dataclass(kw_only=True)
class FBDNoTraits(Scenario):
    birth_rate: SkylineArray
    death_rate: SkylineArray

    @property
    def targets(self) -> list[SkylineTarget]:
        return [
            SkylineTarget(id="birthRateSP", skyline=self.birth_rate),
            SkylineTarget(id="deathRateSP", skyline=self.death_rate),
        ]


def _get_scenario(birth_rate: SkylineArray, death_rate: SkylineArray) -> FBDNoTraits:
    model = get_canonical_model(
        init_state="X",
        states=["X"],
        sampling_rates=FBD_SAMPLING_RATE,
        remove_after_sampling=False,
        birth_rates=SkylineParameter(birth_rate.tolist(), FBD_CHANGE_TIMES),
        death_rates=SkylineParameter(death_rate.tolist(), FBD_CHANGE_TIMES),
    )
    model.add_event(
        TimedEvent(time=FBD_MAX_TIME, firings=1.0, fn=Sampling(removal=True))
    )

    return FBDNoTraits(
        name="fbd-no-traits",
        model=model,
        max_time=FBD_MAX_TIME,
        birth_rate=birth_rate,
        death_rate=death_rate,
        beast_static_data={
            "processLength": FBD_MAX_TIME,
            "changeTimes": " ".join(map(str, FBD_CHANGE_TIMES)),
            **get_prior_params("birthRate", FBD_RATE_UPPER, N_TIME_BINS),
            **get_prior_params("deathRate", FBD_RATE_UPPER, N_TIME_BINS),
            "samplingRate": FBD_SAMPLING_RATE,
            "timePredictor": " ".join(map(str, np.linspace(0, 1, N_TIME_BINS))),
        },
        beast_sample_data={
            "randomPredictor": partial(
                get_random_time_series_predictor, n_time_bins=N_TIME_BINS
            )
        },
    )


FBD_NO_TRAITS_SCENARIOS = {
    f"fbd-no-traits_{i}": _get_scenario(birth_rate, death_rate)
    for i, (birth_rate, death_rate) in enumerate(
        [
            (np.array([0.2] * 10), np.array([0.1] * 10)),
            (
                np.linspace(0.4, 0.1, 10, dtype=np.float64),
                np.linspace(0.1, 0.2, 10, dtype=np.float64),
            ),
            (
                np.array([0.4] * 5 + [0.1] * 3 + [0.01] * 2),
                np.array([0.05] * 7 + [0.3] * 1 + [0.01] * 2),
            ),
        ],
        start=1,
    )
}

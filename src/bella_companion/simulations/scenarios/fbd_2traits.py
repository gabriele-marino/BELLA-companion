from dataclasses import dataclass
from itertools import product

import numpy as np
from phylogenie.skyline import SkylineVector
from phylogenie.treesimulator import Sampling, TimedEvent
from phylogenie.treesimulator.open_population import get_canonical_model

from bella_companion.simulations.scenarios.common import (
    FBD_CHANGE_TIMES,
    FBD_MAX_TIME,
    FBD_RATE_UPPER,
    FBD_SAMPLING_RATE,
    N_TIME_BINS,
    get_prior_params,
    get_start_type_prior_probabilities,
)
from bella_companion.targets import SkylineTarget
from bella_companion.typings import Scenario, SkylineArray


@dataclass(kw_only=True)
class FBD2Traits(Scenario):
    birth_rate_trait1_unset: SkylineArray
    birth_rate_trait1_set: SkylineArray
    death_rate_trait1_unset: SkylineArray
    death_rate_trait1_set: SkylineArray

    def birth_rate(self, trait1: int) -> SkylineArray:
        return self.birth_rate_trait1_set if trait1 else self.birth_rate_trait1_unset

    @property
    def birth_rates(self) -> list[SkylineTarget]:
        return [
            SkylineTarget("birthRateSP", skyline=self.birth_rate(t1), state=f"{t1}{t2}")
            for t1, t2 in product([0, 1], repeat=2)
        ]

    def death_rate(self, trait1: int) -> SkylineArray:
        return self.death_rate_trait1_set if trait1 else self.death_rate_trait1_unset

    @property
    def death_rates(self) -> list[SkylineTarget]:
        return [
            SkylineTarget("deathRateSP", skyline=self.death_rate(t1), state=f"{t1}{t2}")
            for t1, t2 in product([0, 1], repeat=2)
        ]

    @property
    def targets(self) -> list[SkylineTarget]:
        return self.birth_rates + self.death_rates


_STATES = ["00", "01", "10", "11"]
_INIT_STATE = "00"
_N_STATES = len(_STATES)
_BIRTH_RATE_TRAIT1_UNSET = np.linspace(0.6, 0.1, N_TIME_BINS, dtype=np.float64)
_BIRTH_RATE_TRAIT1_SET = np.linspace(0.2, 0.5, N_TIME_BINS, dtype=np.float64)
_DEATH_RATE_TRAIT1_UNSET = np.array([0.1] * 4 + [1.0] * 4 + [0.1] * 2)
_DEATH_RATE_TRAIT1_SET = np.array([0.1] * 4 + [0.5] * 4 + [0.1] * 2)
_DEATH_PREDICTOR = [0] * 4 + [1] * 4 + [0] * 2
_MIGRATION_RATES = (
    0.1 * np.array([[1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1]])
).tolist()

_MODEL = get_canonical_model(
    init_state=_INIT_STATE,
    states=_STATES,
    sampling_rates=FBD_SAMPLING_RATE,
    remove_after_sampling=False,
    birth_rates=SkylineVector(
        value=np.array(
            [
                _BIRTH_RATE_TRAIT1_UNSET,
                _BIRTH_RATE_TRAIT1_UNSET,
                _BIRTH_RATE_TRAIT1_SET,
                _BIRTH_RATE_TRAIT1_SET,
            ]
        ).T.tolist(),
        change_times=FBD_CHANGE_TIMES,
    ),
    death_rates=SkylineVector(
        value=np.array(
            [
                _DEATH_RATE_TRAIT1_UNSET,
                _DEATH_RATE_TRAIT1_UNSET,
                _DEATH_RATE_TRAIT1_SET,
                _DEATH_RATE_TRAIT1_SET,
            ]
        ).T.tolist(),
        change_times=FBD_CHANGE_TIMES,
    ),
    migration_rates=_MIGRATION_RATES,
)
_MODEL.add_event(TimedEvent(time=FBD_MAX_TIME, firings=1.0, fn=Sampling(removal=True)))

FBD_2TRAITS = FBD2Traits(
    name="fbd-2traits",
    model=_MODEL,
    max_time=FBD_MAX_TIME,
    birth_rate_trait1_unset=_BIRTH_RATE_TRAIT1_UNSET,
    birth_rate_trait1_set=_BIRTH_RATE_TRAIT1_SET,
    death_rate_trait1_unset=_DEATH_RATE_TRAIT1_UNSET,
    death_rate_trait1_set=_DEATH_RATE_TRAIT1_SET,
    beast_static_data={
        "types": ",".join(_STATES),
        "startTypePriorProbs": get_start_type_prior_probabilities(_STATES, _INIT_STATE),
        "processLength": FBD_MAX_TIME,
        "changeTimes": " ".join(map(str, FBD_CHANGE_TIMES)),
        **get_prior_params("birthRate", FBD_RATE_UPPER, N_TIME_BINS * _N_STATES),
        **get_prior_params("deathRate", FBD_RATE_UPPER, N_TIME_BINS * _N_STATES),
        "samplingRate": FBD_SAMPLING_RATE,
        "migrationRate": " ".join(map(str, np.array(_MIGRATION_RATES).flatten())),
        "timePredictor": " ".join(
            list(map(str, np.repeat(np.linspace(0, 1, N_TIME_BINS), _N_STATES)))
        ),
        "deathPredictor": " ".join(map(str, np.repeat(_DEATH_PREDICTOR, _N_STATES))),
        "trait1Predictor": " ".join(map(str, [1, 1, 0, 0] * N_TIME_BINS)),
        "trait2Predictor": " ".join(map(str, [0, 1, 0, 1] * N_TIME_BINS)),
    },
)

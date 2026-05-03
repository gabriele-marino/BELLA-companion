from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from phylogenie.treesimulator.open_population import get_epidemiological_model

from bella_companion.simulations.scenarios.common import (
    BECOME_UNINFECTIOUS_RATE,
    EPI_MAX_TIME,
    EPI_SAMPLING_PROPORTION,
    MIGRATION_RATE_UPPER,
    get_last_sample_time,
    get_prior_params,
    get_start_type_prior_probabilities,
)
from bella_companion.targets import MatrixTarget
from bella_companion.typings import Scenario, StateMatrix


@dataclass(kw_only=True)
class EpiMultitype(Scenario):
    migration_predictor: StateMatrix
    migration_rates: StateMatrix

    @property
    def targets(self) -> list[MatrixTarget]:
        return [
            MatrixTarget(
                id="migrationRateSP", states=_STATES, state_matrix=self.migration_rates
            )
        ]


def _get_random_predictor(rng: Generator) -> str:
    predictor = rng.uniform(-1, 1, _N_STATE_PAIRS).tolist()
    return " ".join(map(str, predictor))


_STATES = ["A", "B", "C", "D", "E"]
_REPRODUCTION_NUMBERS = [0.8, 1.0, 1.2, 1.4, 1.6]
_INIT_STATE = "C"
_N_STATES = len(_STATES)
_N_STATE_PAIRS = _N_STATES * (_N_STATES - 1)
_MIGRATION_PREDICTOR = np.random.default_rng(42).uniform(
    -1, 1, (_N_STATES, _N_STATES - 1)
)
_MIGRATION_SIGMOID_AMPLITUDE = 0.04
_MIGRATION_SIGMOID_SCALE = -8
_MIGRATION_RATES = _MIGRATION_SIGMOID_AMPLITUDE / (
    1 + np.exp(_MIGRATION_SIGMOID_SCALE * _MIGRATION_PREDICTOR)
)

EPI_MULTITYPE = EpiMultitype(
    name="epi-multitype",
    model=get_epidemiological_model(
        init_state=_INIT_STATE,
        states=_STATES,
        sampling_proportions=EPI_SAMPLING_PROPORTION,
        reproduction_numbers=_REPRODUCTION_NUMBERS,
        become_uninfectious_rates=BECOME_UNINFECTIOUS_RATE,
        migration_rates=_MIGRATION_RATES.tolist(),
    ),
    max_time=EPI_MAX_TIME,
    migration_predictor=_MIGRATION_PREDICTOR,
    migration_rates=_MIGRATION_RATES,
    beast_static_data={
        "types": ",".join(_STATES),
        "startTypePriorProbs": get_start_type_prior_probabilities(_STATES, _INIT_STATE),
        "processLength": EPI_MAX_TIME,
        **get_prior_params("migrationRate", MIGRATION_RATE_UPPER, _N_STATE_PAIRS),
        "reproductionNumber": " ".join(map(str, _REPRODUCTION_NUMBERS)),
        "becomeUninfectiousRate": BECOME_UNINFECTIOUS_RATE,
        "samplingProportion": EPI_SAMPLING_PROPORTION,
        "migrationPredictor": " ".join(map(str, _MIGRATION_PREDICTOR.flatten())),
    },
    beast_sample_data={"randomPredictor": _get_random_predictor},
    beast_tree_data={"lastSampleTime": get_last_sample_time},
)

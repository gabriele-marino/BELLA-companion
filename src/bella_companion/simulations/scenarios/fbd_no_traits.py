from functools import partial

import numpy as np
from phylogenie.skyline import SkylineParameter
from phylogenie.treesimulator import Sampling, TimedEvent
from phylogenie.treesimulator.open_population import get_canonical_model

from bella_companion.simulations.scenarios.globals import (
    FBD_MAX_TIME,
    FBD_RATE_UPPER,
    FBD_SAMPLING_RATE,
)
from bella_companion.simulations.scenarios.scenario import Scenario
from bella_companion.simulations.scenarios.utils import (
    get_prior_params,
    get_random_time_series_predictor,
)


def _get_scenario(rates: dict[str, list[float]]) -> Scenario:
    if len(rates["birth"]) != len(rates["death"]):
        raise ValueError("Birth rate and death rate lists must have the same length.")
    n_time_bins = len(rates["birth"])
    change_times = np.linspace(0, FBD_MAX_TIME, n_time_bins + 1)[1:-1].tolist()

    model = get_canonical_model(
        init_state="X",
        states=["X"],
        sampling_rates=FBD_SAMPLING_RATE,
        remove_after_sampling=False,
        birth_rates=SkylineParameter(rates["birth"], change_times),
        death_rates=SkylineParameter(rates["death"], change_times),
    )
    model.add_event(
        TimedEvent(time=FBD_MAX_TIME, firings=1.0, fn=Sampling(removal=True))
    )

    return Scenario(
        model=model,
        max_time=FBD_MAX_TIME,
        targets={
            f"{rate}Rate": {f"{rate}RateSPi{i}": values[i] for i in range(n_time_bins)}
            for rate, values in rates.items()
        },
        beast_configs="fbd-no-traits",
        beast_args={
            "processLength": FBD_MAX_TIME,
            "changeTimes": " ".join(map(str, change_times)),
            **get_prior_params("birthRate", FBD_RATE_UPPER, n_time_bins),
            **get_prior_params("deathRate", FBD_RATE_UPPER, n_time_bins),
            "samplingRate": FBD_SAMPLING_RATE,
            "timePredictor": " ".join(map(str, np.linspace(0, 1, n_time_bins))),
        },
        get_random_predictor=partial(
            get_random_time_series_predictor, n_time_bins=n_time_bins
        ),
    )


RATES = [
    {"birth": [0.2] * 10, "death": [0.1] * 10},
    {
        "birth": np.linspace(0.4, 0.1, 10).tolist(),
        "death": np.linspace(0.1, 0.2, 10).tolist(),
    },
    {
        "birth": [0.4] * 5 + [0.1] * 3 + [0.01] * 2,
        "death": [0.05] * 7 + [0.3] * 1 + [0.01] * 2,
    },
]
FBD_NO_TRAITS_SCENARIOS = [_get_scenario(r) for r in RATES]

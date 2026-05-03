from typing import Any

import numpy as np
from numpy.random import Generator
from phylogenie import TreeNode

EPI_MAX_TIME = 250
EPI_SAMPLING_PROPORTION = 0.15
BECOME_UNINFECTIOUS_RATE = 0.07
REPRODUCTION_NUMBER_UPPER = 5
MIGRATION_RATE_UPPER = 0.2

FBD_MAX_TIME = 35
FBD_SAMPLING_RATE = 0.2
FBD_RATE_UPPER = 2


def _get_change_times(max_time: float, n_time_bins: int) -> list[float]:
    return np.linspace(0, max_time, n_time_bins + 1)[1:-1].tolist()


N_TIME_BINS = 10
FBD_CHANGE_TIMES = _get_change_times(FBD_MAX_TIME, N_TIME_BINS)
FBD_TIME_AXIS = np.linspace(0, FBD_MAX_TIME, N_TIME_BINS + 1, dtype=np.float64)
EPI_CHANGE_TIMES = _get_change_times(EPI_MAX_TIME, N_TIME_BINS)
EPI_TIME_AXIS = np.linspace(0, EPI_MAX_TIME, N_TIME_BINS + 1, dtype=np.float64)


def get_start_type_prior_probabilities(types: list[str], init_type: str):
    start_type_prior_probabilities = ["0"] * len(types)
    start_type_prior_probabilities[types.index(init_type)] = "1"
    return " ".join(start_type_prior_probabilities)


def get_random_time_series_predictor(rng: Generator, n_time_bins: int) -> str:
    predictor = np.cumsum(rng.normal(size=n_time_bins)).tolist()
    return " ".join(map(str, predictor))


def get_prior_params(target: str, upper: float, n: int) -> dict[str, Any]:
    return {
        f"{target}Upper": upper,
        f"{target}Init": " ".join([str(upper / 2)] * n),
    }


def get_last_sample_time(tree: TreeNode) -> float:
    return tree.origin

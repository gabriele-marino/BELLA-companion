from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from numpy.random import Generator
from phylogenie.treesimulator import UnboundedPopulationEvent


class ScenarioType(Enum):
    EPI = "epi"
    FBD = "fbd"


@dataclass
class Scenario:
    type: ScenarioType
    max_time: float
    events: list[UnboundedPopulationEvent]
    init_state: str
    get_random_predictor: Callable[[Generator], list[float]]
    beast_args: dict[str, Any]
    targets: dict[str, dict[str, float]]

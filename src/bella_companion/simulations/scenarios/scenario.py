from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from numpy.random import Generator
from phylogenie import TreeNode
from phylogenie.treesimulator import Model


@dataclass
class Scenario:
    model: Model
    max_time: float | None
    targets: dict[str, dict[str, float]]
    beast_configs: str
    beast_args: dict[str, Any]
    get_random_predictor: Callable[[Generator], list[float]] | None = None
    tree_beast_args: dict[str, Callable[[TreeNode], Any]] | None = None

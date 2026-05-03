from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from jaxtyping import Float
from numpy.random import Generator
from phylogenie import TreeNode
from phylogenie.treesimulator import Model as PhylogenieModel

BeastDataKey: TypeAlias = str
BeastDataValue: TypeAlias = str | float
SampleDataFn: TypeAlias = Callable[[Generator], BeastDataValue]
TreeDataFn: TypeAlias = Callable[[TreeNode], BeastDataValue]

ScenarioID: TypeAlias = str
ModelID: TypeAlias = str
LogID: TypeAlias = str
JobID: TypeAlias = str
JobBatch: TypeAlias = dict[LogID, JobID]
ModelJobBatch: TypeAlias = dict[ModelID, JobBatch]

TargetID: TypeAlias = str
TargetKey: TypeAlias = str
TargetValue: TypeAlias = float
BayesMLPID: TypeAlias = str


class JobMetadata(TypedDict):
    status: str
    total_hours: float


@dataclass
class Target(ABC):
    id: TargetID

    @property
    @abstractmethod
    def value_map(self) -> Mapping[TargetKey, TargetValue]: ...

    @property
    def keys(self) -> list[TargetKey]:
        return list(self.value_map.keys())

    @property
    def values(self) -> list[TargetValue]:
        return list(self.value_map.values())


@dataclass(kw_only=True)
class Scenario:
    name: str
    model: PhylogenieModel
    max_time: float
    beast_static_data: Mapping[BeastDataKey, BeastDataValue]
    beast_sample_data: Mapping[BeastDataKey, SampleDataFn] | None = None
    beast_tree_data: Mapping[BeastDataKey, TreeDataFn] | None = None

    @property
    @abstractmethod
    def targets(self) -> Iterable[Target]: ...


@dataclass(kw_only=True)
class BELLAConfig:
    weights_prior: Literal["Normal", "Laplace"]
    hidden_activation: Literal["Identity", "Sigmoid", "ReLU", "Softplus", "Tanh"]
    nodes: list[int]

    def get_beast_data(self) -> dict[BeastDataKey, BeastDataValue]:
        return {
            "nodes": " ".join(map(str, self.nodes)),
            "layersRange": ",".join(map(str, range(len(self.nodes) + 1))),
            "weightsPrior": self.weights_prior,
            "hiddenActivation": self.hidden_activation,
        }


ArrayLike: TypeAlias = np.typing.ArrayLike
Array: TypeAlias = np.typing.NDArray[np.float64]

SkylineArray: TypeAlias = Float[Array, "n_timebins"]  # noqa: F821
StateMatrix: TypeAlias = Float[Array, "n_states n_states-1"]  # noqa: F722

LayerWeights: TypeAlias = Float[Array, "_n_nodes_in _n_nodes_out"]  # noqa: F722
Weights: TypeAlias = Sequence[LayerWeights]
PosteriorWeights: TypeAlias = Sequence[Weights]
PosteriorWeightsByMLP: TypeAlias = Mapping[BayesMLPID, PosteriorWeights]

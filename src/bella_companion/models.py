from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, Type

from bella_companion.backend import ActivationFunction


class WeightsPrior(str, Enum):
    NORMAL = "Normal"
    LAPLACE = "Laplace"


@dataclass
class BELLASetting:
    weights_prior: WeightsPrior
    hidden_activation: Type[ActivationFunction]
    nodes: Sequence[int]

    def get_beast_data(self) -> dict[str, Any]:
        return {
            "nodes": " ".join(map(str, self.nodes)),
            "layersRange": ",".join(map(str, range(len(self.nodes) + 1))),
            "weightsPrior": self.weights_prior.value,
            "hiddenActivation": self.hidden_activation.__name__,
        }

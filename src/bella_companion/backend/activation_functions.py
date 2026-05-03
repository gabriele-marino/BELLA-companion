from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from bella_companion.typings import Array


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: ArrayLike) -> Array: ...


class Identity(ActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.asarray(x, dtype=np.float64)


class Sigmoid(ActivationFunction):
    def __init__(
        self,
        lower: float = 0.0,
        upper: float = 1.0,
        shape: float = 1.0,
        midpoint: float = 0.0,
    ):
        self._lower = lower
        self._upper = upper
        self._shape = shape
        self._midpoint = midpoint

    def __call__(self, x: ArrayLike) -> Array:
        x = np.asarray(x, dtype=np.float64)
        return self._lower + (self._upper - self._lower) / (
            1 + np.exp(-self._shape * (x - self._midpoint))
        )


class ReLU(ActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.maximum(0, x)


class Softplus(ActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.log1p(np.exp(x))


class Tanh(ActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.tanh(x)


_ACTIVATION_FUNCTIONS_REGISTRY = {
    class_.__name__.lower(): class_ for class_ in ActivationFunction.__subclasses__()
}
ActivationFunctionLike = str | ActivationFunction


def as_activation_function(activation: ActivationFunctionLike) -> ActivationFunction:
    if isinstance(activation, str):
        return _ACTIVATION_FUNCTIONS_REGISTRY[activation.lower()]()
    return activation

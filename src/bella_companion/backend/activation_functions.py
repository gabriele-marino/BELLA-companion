from abc import abstractmethod
from typing import Callable

import numpy as np
from autoregistry import Registry
from numpy.typing import ArrayLike

from bella_companion.backend.type_hints import Array

ActivationFunction = Callable[[ArrayLike], Array]
ActivationFunctionLike = str | ActivationFunction


class RegisteredActivationFunction(Registry):
    @abstractmethod
    def __call__(self, x: ArrayLike) -> Array: ...


class Identity(RegisteredActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.asarray(x, dtype=np.float64)

    def __repr__(self) -> str:
        return "Identity()"


class Sigmoid(RegisteredActivationFunction):
    def __init__(self, lower: float = 0.0, upper: float = 1.0, shape: float = 1.0):
        self._lower = lower
        self._upper = upper
        self._shape = shape

    def __call__(self, x: ArrayLike) -> Array:
        x = np.asarray(x, dtype=np.float64)
        return self._lower + (self._upper - self._lower) / (
            1 + np.exp(-self._shape * x)
        )

    def __repr__(self) -> str:
        return f"Sigmoid(lower={self._lower}, upper={self._upper}, shape={self._shape})"


class ReLU(RegisteredActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.maximum(0, x)

    def __repr__(self) -> str:
        return "ReLU()"


class Softplus(RegisteredActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.log1p(np.exp(x))

    def __repr__(self) -> str:
        return "Softplus()"


class Tanh(RegisteredActivationFunction):
    def __call__(self, x: ArrayLike) -> Array:
        return np.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


def as_activation_function(
    activation: ActivationFunctionLike,
) -> ActivationFunction:
    if isinstance(activation, str):
        return RegisteredActivationFunction[activation]()
    return activation

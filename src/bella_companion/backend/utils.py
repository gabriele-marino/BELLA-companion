import numpy as np
from numpy.typing import ArrayLike

from bella_companion.typings import Array


def normalize(array: ArrayLike, axis: int | None = None) -> Array:
    """Perform min-max normalization on the input array along the specified axis, scaling values to the [0, 1] range."""
    return (array - np.min(array, axis=axis)) / (
        np.max(array, axis=axis) - np.min(array, axis=axis)
    )

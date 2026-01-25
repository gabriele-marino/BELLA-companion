import numpy as np
from numpy.typing import ArrayLike

from bella_companion.backend.type_hints import Array


def normalize(array: ArrayLike, axis: int | None = None) -> Array:
    """
    Normalize an array to the range [0, 1].

    Parameters
    ----------
    array : ArrayLike
        Input array to be normalized.
    axis : int | None, optional
        Axis along which to normalize. If None, normalize the entire array, by default None.

    Returns
    -------
    Array
        Normalized array with values scaled to [0, 1].
    """
    return (array - np.min(array, axis=axis)) / (
        np.max(array, axis=axis) - np.min(array, axis=axis)
    )

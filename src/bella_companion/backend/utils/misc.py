from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from phylogenie import load_nexus

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


def load_nexus_with_burnin(
    tree_file: str | Path,
    burnin: int | float = 0.1,
):
    """
    Load trees from a Nexus file, applying burn-in to remove initial samples.

    Parameters
    ----------
    tree_file : str | Path
        Path to the Nexus file containing the trees.
    burnin : int | float, optional
        Burn-in proportion (float between 0 and 1) or number of trees (int) to discard from the start, by default 0.1.

    Returns
    -------
    list[Tree]
        List of trees after applying burn-in.
    """
    all_trees = list(load_nexus(tree_file).values())
    if isinstance(burnin, float):
        burnin = int(len(all_trees) * burnin)
    return all_trees[burnin:]

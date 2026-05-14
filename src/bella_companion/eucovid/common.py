import numpy as np

from bella_companion.typings import Array


def transform(x: Array) -> Array:
    x_log = np.log(x + 1)
    return (x_log - x_log.mean()) / x_log.std(ddof=1)

from collections.abc import Iterable
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

Color: TypeAlias = (
    str
    | np.typing.NDArray[np.floating]
    | tuple[float, float, float]
    | tuple[float, float, float, float]
)


def skyline_plot(
    data: ArrayLike,
    x: ArrayLike | None = None,
    ax: Axes | None = None,
    step_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Skyline (step) plot.

    Args:
        data: The y values for the skyline plot, of shape (n_points,).
        x: The x values, of shape (n_points + 1,). If None, uses indices.
            If provided, the first x value corresponds to the start of the first step,
            and the last x value corresponds to the end of the last step.
        ax: The matplotlib Axes to plot on. If None, uses the current Axes.
        step_kwargs: Additional keyword arguments for the step plot.

    Returns:
        The matplotlib Axes with the plot.
    """
    data = np.asarray(data, dtype=np.float64)
    data = [data[0], *data]

    if ax is None:
        ax = plt.gca()
    if x is None:
        x = list(range(len(data)))
    if step_kwargs is None:
        step_kwargs = {}

    ax.step(x, data, **step_kwargs)  # pyright: ignore
    return ax


def ribbon_plot(
    y: ArrayLike,
    x: ArrayLike | None = None,
    color: Color | None = None,
    label: str | None = None,
    ax: Axes | None = None,
    skyline: bool = False,
    percentiles: Iterable[float] = (50, 95),
    show_fill: bool = True,
    fill_kwargs: dict[str, Any] | None = None,
    show_samples: bool = True,
    samples_kwargs: dict[str, Any] | None = None,
    show_median: bool = True,
    median_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Ribbon plot with uncertainty intervals.

    Args:
        y: The y values, of shape (n_samples, n_points).
        x: The x values. If None, uses indices.
            If skyline is False, x should have shape (n_points,).
            If skyline is True, x should have shape (n_points + 1,),
            where the first x value corresponds to the start of the first step,
            the last x value corresponds to the end of the last step.
        color: The color for the plot.
        label: The label for the median line.
        ax: The matplotlib Axes to plot on. If None, uses the current Axes.
        skyline: Whether to use a skyline (step) plot.
        percentiles: The percentiles for the percentile interval fill.
        show_fill: Whether to show the percentile interval fill.
        fill_kwargs: Additional keyword arguments for the fill_between call.
        show_samples: Whether to show individual sample lines.
        samples_kwargs: Additional keyword arguments for the sample lines.
        show_median: Whether to show the median line.
        median_kwargs: Additional keyword arguments for the median line.

    Returns:
        The matplotlib Axes with the plot.
    """
    if ax is None:
        ax = plt.gca()

    y = np.asarray(y, dtype=np.float64)
    if x is None:
        _, n_points = y.shape
        if skyline:
            n_points += 1
        x = list(range(n_points))

    if show_fill:
        for percentile in percentiles:
            lower = np.percentile(y, 50 - percentile / 2, axis=0)
            high = np.percentile(y, 50 + percentile / 2, axis=0)
            if fill_kwargs is None:
                fill_kwargs = {}
            fill_kwargs["alpha"] = 0.20 + (95 - percentile) * (0.30 / 95)
            if "color" not in fill_kwargs:
                fill_kwargs["color"] = color
            if skyline:
                fill_kwargs["step"] = "pre"
                lower = [lower[0], *lower]
                high = [high[0], *high]
            ax.fill_between(x, lower, high, **fill_kwargs)  # pyright: ignore

    if show_samples:
        if samples_kwargs is None:
            samples_kwargs = {}
        if "alpha" not in samples_kwargs:
            samples_kwargs["alpha"] = 0.25
        if "color" not in samples_kwargs:
            samples_kwargs["color"] = color
        for sample_y in y:
            if skyline:
                skyline_plot(data=sample_y, x=x, ax=ax, step_kwargs=samples_kwargs)
            else:
                ax.plot(x, sample_y, **samples_kwargs)  # pyright: ignore

    if show_median:
        median = np.median(y, axis=0)
        if median_kwargs is None:
            median_kwargs = {}
        if "color" not in median_kwargs:
            median_kwargs["color"] = color
        if "label" not in median_kwargs:
            median_kwargs["label"] = label
        if skyline:
            skyline_plot(data=median, x=x, ax=ax, step_kwargs=median_kwargs)
        else:
            ax.plot(x, median, **median_kwargs)  # pyright: ignore

    return ax

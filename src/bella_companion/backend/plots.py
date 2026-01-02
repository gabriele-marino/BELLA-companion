from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

Color = (
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
    """
    Plot a skyline (step) plot.

    Parameters
    ----------
    data : ArrayLike
        The y values for the skyline plot, of shape (n_points,).
    x : ArrayLike | None, optional
        The x values, of shape (n_points + 1,), by default None (uses indices).
        The first x value corresponds to the start of the first step, the
        last x value corresponds to the end of the last step.
    ax : Axes | None, optional
        The matplotlib Axes to plot on, by default None (uses current Axes).
    step_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments for the step plot, by default None.

    Returns
    -------
    Axes
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

    ax.step(x, data, **(step_kwargs or {}))  # pyright: ignore
    return ax


def ribbon_plot(
    y: ArrayLike,
    x: ArrayLike | None = None,
    color: Color | None = None,
    label: str | None = None,
    ax: Axes | None = None,
    skyline: bool = False,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
    show_fill: bool = True,
    fill_kwargs: dict[str, Any] | None = None,
    show_samples: bool = True,
    samples_kwargs: dict[str, Any] | None = None,
    show_median: bool = True,
    median_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """
    Plot a ribbon plot with uncertainty intervals.

    Parameters
    ----------
    y : ArrayLike
        The y values, of shape (n_samples, n_points).
    x : ArrayLike | None, optional
        The x values, by default None (uses indices).
        If skyline is True, x should have shape (n_points + 1,),
        where the first x value corresponds to the start of the first step,
        the last x value corresponds to the end of the last step.
        If skyline is False, x should have shape (n_points,).
    color : str | None, optional
        The color for the plot, by default None.
    label : str | None, optional
        The label for the median line, by default None.
    ax : Axes | None, optional
        The matplotlib Axes to plot on, by default None (uses current Axes).
    skyline : bool, optional
        Whether to use a skyline (step) plot, by default False.
    lower_percentile : float, optional
        The lower percentile for the percentile interval, by default 2.5.
    upper_percentile : float, optional
        The upper percentile for the percentile interval, by default 97.5.
    show_fill : bool, optional
        Whether to show the percentile interval fill, by default True.
    fill_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments for the fill_between call, by default None.
    show_samples : bool, optional
        Whether to show individual sample lines, by default True.
    samples_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments for the sample lines, by default None.
    show_median : bool, optional
        Whether to show the median line, by default True.
    median_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments for the median line, by default None.

    Returns
    -------
    Axes
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
        lower = np.percentile(y, lower_percentile, axis=0)
        high = np.percentile(y, upper_percentile, axis=0)
        if fill_kwargs is None:
            fill_kwargs = {}
        if "alpha" not in fill_kwargs:
            fill_kwargs["alpha"] = 0.25
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

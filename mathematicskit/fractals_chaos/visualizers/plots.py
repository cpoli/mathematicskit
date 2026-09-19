"""Plotting helpers for mathematicskit.fractals_chaos: escape-time heatmaps,
box-counting log-log fits, IFS point clouds, and cellular-automaton
space-time diagrams.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_escape_time", "plot_box_counting", "plot_ifs_points", "plot_ca_spacetime"]


def plot_escape_time(result, ax=None, cmap: str = "viridis"):
    """Heatmap of an escape-time grid (Mandelbrot/Julia set).

    Parameters
    ----------
    result : EscapeTimeResult
        From :func:`~mathematicskit.fractals_chaos.systems.mandelbrot_julia.mandelbrot_set`
        or :func:`~mathematicskit.fractals_chaos.systems.mandelbrot_julia.julia_set`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    re_min, re_max, im_min, im_max = result.extent
    ax.imshow(result.iterations, extent=(re_min, re_max, im_min, im_max), origin="lower", cmap=cmap)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Escape time (max_iter={result.max_iter})")
    return ax


def plot_box_counting(result, ax=None, label=None):
    """Log-log plot of box count vs. inverse box size, with the fitted slope.

    Parameters
    ----------
    result : BoxCountingResult
        From :func:`~mathematicskit.fractals_chaos.systems.box_counting.box_counting_dimension`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    inv_size = 1.0 / result.box_sizes
    ax.loglog(inv_size, result.box_counts, "o", label=label or f"D ~ {result.dimension:.3f}")
    ax.set_xlabel("1 / box size")
    ax.set_ylabel("box count")
    ax.set_title("Box-counting dimension")
    ax.legend()
    return ax


def plot_ifs_points(points: np.ndarray, ax=None, s: float = 0.2, color: str = "forestgreen"):
    """Scatter plot of an iterated function system's generated point cloud.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
        From :meth:`~mathematicskit.fractals_chaos.core.base.IteratedFunctionSystem.generate`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    s : float
        Marker size.
    color : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=s, color=color, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_ca_spacetime(history: np.ndarray, ax=None, cmap: str = "binary"):
    """Space-time diagram of a 1D cellular automaton's history (one row per generation).

    Parameters
    ----------
    history : ndarray, shape (n_steps + 1, width)
        From :meth:`~mathematicskit.fractals_chaos.core.base.CellularAutomaton.run`
        on an :class:`~mathematicskit.fractals_chaos.systems.cellular_automata.ElementaryCA`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(history, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xlabel("cell")
    ax.set_ylabel("generation")
    ax.set_title("Cellular automaton space-time diagram")
    return ax

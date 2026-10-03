"""Plotting helpers for mathematicskit.fractals_chaos: escape-time heatmaps,
box-counting log-log fits, IFS point clouds, cellular-automaton
space-time diagrams, and an animation of the Game of Life.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["plot_escape_time", "plot_box_counting", "plot_ifs_points", "plot_ca_spacetime", "animate_life"]


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


def animate_life(history: np.ndarray, interval: int = 120, cmap: str = "binary"):
    """Animate a 2D cellular automaton's history, one generation per frame.

    Parameters
    ----------
    history : ndarray, shape (n_steps + 1, ny, nx)
        From :meth:`~mathematicskit.fractals_chaos.core.base.CellularAutomaton.run`
        on a :class:`~mathematicskit.fractals_chaos.systems.cellular_automata.GameOfLife`.
    interval : int
        Delay between frames, in milliseconds.
    cmap : str

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from mathematicskit.fractals_chaos import GameOfLife
    >>> grid = np.zeros((8, 8), dtype=np.int64)
    >>> grid[1, 2] = grid[2, 3] = 1
    >>> grid[3, 1:4] = 1  # a glider
    >>> anim = animate_life(GameOfLife(grid).run(4))
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    history = np.asarray(history)
    if history.ndim != 3:
        raise ValueError("history must have shape (n_steps + 1, ny, nx)")
    fig, ax = plt.subplots(figsize=(6, 6 * history.shape[1] / history.shape[2] + 0.4))
    image = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    fig.tight_layout()

    def update(k):
        image.set_data(history[k])
        title.set_text(f"generation {k}: {int(history[k].sum())} live cells")
        return [image, title]

    return FuncAnimation(fig, update, frames=len(history), interval=interval)

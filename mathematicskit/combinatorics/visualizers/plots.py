"""Plotting helpers for mathematicskit.combinatorics: Pascal's triangle, Young
diagrams, and partition-count growth."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_pascals_triangle", "plot_young_diagram", "plot_partition_counts"]


def plot_pascals_triangle(triangle, ax=None):
    """Heatmap of Pascal's triangle (left-justified, zero-padded).

    Parameters
    ----------
    triangle : list of list of int
        From :func:`~mathematicskit.combinatorics.systems.pascals_triangle.pascals_triangle`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    n_rows = len(triangle)
    grid = np.zeros((n_rows, n_rows))
    for i, row in enumerate(triangle):
        grid[i, : len(row)] = row
    masked = np.ma.masked_where(grid == 0, grid)
    ax.imshow(masked, cmap="viridis")
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_title("Pascal's triangle")
    return ax


def plot_young_diagram(diagram, ax=None):
    """Draw a Young/Ferrers diagram as a grid of unit squares.

    Parameters
    ----------
    diagram : YoungDiagram
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for i, part in enumerate(diagram.parts):
        for j in range(part):
            ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, edgecolor="black", facecolor="steelblue"))
    max_part = max(diagram.parts) if diagram.parts else 1
    ax.set_xlim(0, max_part)
    ax.set_ylim(-len(diagram.parts), 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Young diagram of {diagram.parts}")
    return ax


def plot_partition_counts(max_n: int, ax=None):
    """Semilog plot of the partition function ``p(n)`` for ``n = 0..max_n``.

    Parameters
    ----------
    max_n : int
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from mathematicskit.combinatorics.systems.partitions import partition_function

    if ax is None:
        _, ax = plt.subplots()
    ns = np.arange(0, max_n + 1)
    counts = np.array([partition_function(int(n)) for n in ns])
    ax.semilogy(ns, counts, "o-")
    ax.set_xlabel("n")
    ax.set_ylabel("p(n)")
    ax.set_title("Integer partition function")
    return ax

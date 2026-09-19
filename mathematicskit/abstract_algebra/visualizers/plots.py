"""Plotting helpers for mathematicskit.abstract_algebra: Cayley table heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt

__all__ = ["plot_cayley_table"]


def plot_cayley_table(group, ax=None, cmap: str = "tab20"):
    """Heatmap of a finite group's Cayley table.

    Parameters
    ----------
    group : FiniteGroup
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    table = group.cayley_table()
    ax.imshow(table, cmap=cmap)
    ax.set_xlabel("element index")
    ax.set_ylabel("element index")
    ax.set_title(f"Cayley table (order {group.order})")
    return ax

"""Plotting helpers for mathkit.optimization: 2D contour + iterate-path
plots, and convergence-rate comparison plots.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_contour_path", "plot_convergence_comparison"]


def plot_contour_path(f: Callable[[np.ndarray], float], result, ax=None, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), n_grid: int = 200, label=None):
    """Contour plot of a 2D objective with an optimizer's iterate path overlaid.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float`` (called on 2-vectors).
    result : OptimizeResult
        From any :class:`~mathkit.optimization.core.base.UnconstrainedOptimizer`,
        solved on a 2D problem.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    x_range, y_range : tuple of float
        Contour grid extent.
    n_grid : int
        Grid resolution per axis.
    label : str, optional
        Legend label for the path; defaults to ``result.method``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    xs = np.linspace(*x_range, n_grid)
    ys = np.linspace(*y_range, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    ax.contour(X, Y, Z, levels=30, cmap="Greys", linewidths=0.5)
    path = np.asarray(result.path)
    ax.plot(path[:, 0], path[:, 1], "o-", ms=3, lw=1, label=label or result.method)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimizer iterate path")
    ax.legend()
    return ax


def plot_convergence_comparison(results: dict[str, object], f: Callable[[np.ndarray], float], f_star: float, ax=None):
    """Semilog plot of ``f(x_k) - f*`` vs. iteration, across several methods.

    Parameters
    ----------
    results : dict of str -> OptimizeResult
        E.g. from :func:`~mathkit.optimization.utils.comparison.compare_optimizers`.
    f : callable
    f_star : float
        Known (or best available) optimal value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for name, result in results.items():
        gap = np.array([f(x) - f_star for x in result.path])
        gap = np.where(gap <= 0.0, np.nan, gap)
        ax.semilogy(np.arange(len(gap)), gap, "o-", ms=3, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x_k) - f*")
    ax.set_title("Convergence-rate comparison")
    ax.legend()
    return ax

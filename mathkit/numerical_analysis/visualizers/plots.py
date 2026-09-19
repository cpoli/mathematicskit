"""Plotting helpers for root-finding convergence, interpolants, and the
Runge-phenomenon comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_convergence_history", "plot_interpolant", "plot_runge_phenomenon"]


def plot_convergence_history(result, root_exact=None, ax=None, label=None):
    """Semilog plot of ``|x_n - root|`` vs. iteration, for a :class:`~mathkit.numerical_analysis.core.base.RootResult`.

    Parameters
    ----------
    result : RootResult
        A solved root-finder result (``result.history``).
    root_exact : float, optional
        Reference root to measure error against; defaults to
        ``result.root`` (the sequence's own final iterate).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional
        Legend label; defaults to ``result.method``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    root = result.root if root_exact is None else root_exact
    errors = np.abs(result.history - root)
    errors = np.where(errors == 0.0, np.nan, errors)  # avoid log(0)
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(np.arange(len(errors)), errors, marker="o", ms=3, label=label or result.method)
    ax.set_xlabel("iteration")
    ax.set_ylabel("|x_n - root|")
    ax.set_title(f"Convergence history ({result.method})")
    return ax


def plot_interpolant(interpolant, x_fine=None, ax=None, show_nodes=True):
    """Plot an interpolant's curve alongside its data nodes.

    Parameters
    ----------
    interpolant : Interpolant
        Any concrete :class:`~mathkit.numerical_analysis.core.base.Interpolant`
        (e.g. :class:`~mathkit.numerical_analysis.systems.splines.CubicSpline`).
    x_fine : ndarray, optional
        Evaluation grid; defaults to 400 points spanning the nodes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    show_nodes : bool
        Whether to scatter the original data nodes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if x_fine is None:
        x_fine = np.linspace(interpolant.x.min(), interpolant.x.max(), 400)
    y_fine = interpolant.evaluate(x_fine)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x_fine, y_fine, color="steelblue", label="interpolant")
    if show_nodes:
        ax.scatter(interpolant.x, interpolant.y, color="firebrick", zorder=3, label="nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(type(interpolant).__name__)
    ax.legend()
    return ax


def plot_runge_phenomenon(degrees, equal_errors, chebyshev_errors, ax=None):
    """Semilog plot of max interpolation error vs. degree, equally spaced
    vs. Chebyshev nodes (see
    :func:`mathkit.numerical_analysis.systems.chebyshev.runge_phenomenon_errors`).

    Parameters
    ----------
    degrees : array-like of int
    equal_errors, chebyshev_errors : array-like of float
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(degrees, equal_errors, "o-", color="firebrick", label="equally spaced")
    ax.semilogy(degrees, chebyshev_errors, "o-", color="steelblue", label="Chebyshev")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("max |error|")
    ax.set_title("Runge phenomenon: node placement matters")
    ax.legend()
    return ax

"""Plotting helpers for mathematicskit.calculus: quadrature convergence and
Taylor-series partial sums."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.calculus.utils.series_utils import partial_sums

__all__ = ["plot_quadrature_convergence", "plot_taylor_series_convergence"]


def plot_quadrature_convergence(rule_factory, f, a, b, exact, ns, ax=None, label=None):
    """Log-log plot of quadrature error vs. number of subintervals/nodes.

    Parameters
    ----------
    rule_factory : callable
        ``n -> Quadrature`` (e.g. ``lambda n: SimpsonsRule(n=n)``).
    f : callable
    a, b : float
    exact : float
        Known exact integral value.
    ns : array-like of int
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ns = np.asarray(ns)
    errors = np.array([abs(rule_factory(int(n)).integrate(f, a, b).value - exact) for n in ns])
    errors = np.where(errors == 0.0, np.nan, errors)
    if ax is None:
        _, ax = plt.subplots()
    ax.loglog(ns, errors, "o-", label=label)
    ax.set_xlabel("n")
    ax.set_ylabel("|error|")
    ax.set_title("Quadrature convergence")
    if label:
        ax.legend()
    return ax


def plot_taylor_series_convergence(coefficients: np.ndarray, x: float, exact: float, ax=None, label=None):
    """Semilog plot of Taylor partial-sum error vs. truncation degree.

    Parameters
    ----------
    coefficients : ndarray
        From :func:`mathematicskit.calculus.systems.taylor_series.maclaurin_coefficients`.
    x : float
    exact : float
        The function's true value at ``x``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sums = partial_sums(coefficients, x)
    errors = np.abs(sums - exact)
    errors = np.where(errors == 0.0, np.nan, errors)
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(np.arange(len(errors)), errors, "o-", label=label)
    ax.set_xlabel("truncation degree")
    ax.set_ylabel("|error|")
    ax.set_title("Taylor series convergence")
    if label:
        ax.legend()
    return ax

"""Plotting helpers for mathkit.statistics: boxplots, regression
diagnostic plots, and bootstrap-distribution histograms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_boxplot", "plot_regression_fit", "plot_residuals"]


def plot_boxplot(data: np.ndarray, ax=None, label=None):
    """Boxplot (five-number summary) of a dataset.

    Parameters
    ----------
    data : array-like, shape (n,)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.boxplot(np.asarray(data, dtype=np.float64), tick_labels=[label] if label else None)
    ax.set_ylabel("value")
    ax.set_title("Boxplot")
    return ax


def plot_regression_fit(x: np.ndarray, y: np.ndarray, result, ax=None):
    """Scatter of ``(x, y)`` with the fitted regression line overlaid.

    Only meaningful for a single-predictor fit
    (:func:`~mathkit.statistics.systems.regression.linear_regression`
    with a 1D ``x``).

    Parameters
    ----------
    x, y : array-like, shape (n,)
    result : RegressionResult
        From :func:`~mathkit.statistics.systems.regression.linear_regression`
        (called with ``add_intercept=True``, so ``coefficients =
        [intercept, slope]``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ax.scatter(x, y, color="steelblue", label="data")
    xs = np.linspace(x.min(), x.max(), 100)
    intercept, slope = result.coefficients
    ax.plot(xs, intercept + slope * xs, color="firebrick", label=f"fit (R^2={result.r_squared:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear regression fit")
    ax.legend()
    return ax


def plot_residuals(result, ax=None):
    """Residuals-vs-fitted-values plot, the standard regression diagnostic.

    Parameters
    ----------
    result : RegressionResult
        From :func:`~mathkit.statistics.systems.regression.linear_regression`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(result.fitted_values, result.residuals, color="steelblue")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("fitted values")
    ax.set_ylabel("residuals")
    ax.set_title("Residuals vs. fitted values")
    return ax

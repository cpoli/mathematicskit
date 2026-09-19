"""Smoke tests for mathkit.statistics.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathkit.statistics.systems.regression import linear_regression
from mathkit.statistics.visualizers.plots import plot_boxplot, plot_regression_fit, plot_residuals


def test_plot_boxplot_returns_axes():
    ax = plot_boxplot(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_regression_fit_returns_axes():
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0
    result = linear_regression(x, y)
    ax = plot_regression_fit(x, y, result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_residuals_returns_axes():
    x = np.linspace(0, 10, 20)
    y = 2.0 * x + 1.0
    result = linear_regression(x, y)
    ax = plot_residuals(result)
    assert isinstance(ax, matplotlib.axes.Axes)

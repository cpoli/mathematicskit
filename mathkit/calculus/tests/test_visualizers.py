"""Smoke tests for mathkit.calculus.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathkit.calculus.systems.quadrature import SimpsonsRule
from mathkit.calculus.systems.taylor_series import maclaurin_coefficients
from mathkit.calculus.visualizers.plots import plot_quadrature_convergence, plot_taylor_series_convergence


def test_plot_quadrature_convergence_returns_axes():
    ax = plot_quadrature_convergence(lambda n: SimpsonsRule(n=n), np.sin, 0.0, np.pi, 2.0, [2, 4, 8, 16])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_taylor_series_convergence_returns_axes():
    coeffs = maclaurin_coefficients("exp", order=15)
    ax = plot_taylor_series_convergence(coeffs, 1.0, np.e)
    assert isinstance(ax, matplotlib.axes.Axes)

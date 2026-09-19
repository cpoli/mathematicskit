"""Smoke tests for mathkit.special_functions.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathkit.special_functions.systems.orthogonal_polynomials import legendre_polynomial
from mathkit.special_functions.visualizers.plots import plot_fft_timing_comparison, plot_polynomial_family


def test_plot_polynomial_family_returns_axes():
    ax = plot_polynomial_family(legendre_polynomial, degrees=[0, 1, 2, 3])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_fft_timing_comparison_returns_axes():
    ax = plot_fft_timing_comparison([32, 64, 128])
    assert isinstance(ax, matplotlib.axes.Axes)

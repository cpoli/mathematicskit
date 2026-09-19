"""Smoke tests for mathematicskit.linalg.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.linalg.systems.iterative import ConjugateGradient
from mathematicskit.linalg.utils.matrix_utils import random_spd_matrix
from mathematicskit.linalg.visualizers.plots import plot_eigenvalue_spectrum, plot_matrix_heatmap, plot_residual_history


def test_plot_matrix_heatmap_returns_axes():
    ax = plot_matrix_heatmap(random_spd_matrix(4, seed=40))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_residual_history_returns_axes():
    a = random_spd_matrix(4, seed=41)
    b = np.ones(4)
    result = ConjugateGradient().solve(a, b)
    ax = plot_residual_history(result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_eigenvalue_spectrum_returns_axes():
    ax = plot_eigenvalue_spectrum(np.array([1.0, 2.0, 3.0]))
    assert isinstance(ax, matplotlib.axes.Axes)

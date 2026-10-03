"""Smoke tests for mathematicskit.linalg.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation, PillowWriter

from mathematicskit.linalg.systems.eigen import eigen_symmetric, power_iteration
from mathematicskit.linalg.systems.iterative import ConjugateGradient
from mathematicskit.linalg.utils.matrix_utils import random_spd_matrix
from mathematicskit.linalg.visualizers.plots import animate_power_iteration, plot_eigenvalue_spectrum, plot_matrix_heatmap, plot_residual_history


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


@pytest.mark.parametrize("a", [np.array([[2.0, 1.0], [1.0, 3.0]]), np.diag([4.0, -3.0, 1.0])])
def test_animate_power_iteration_saves(tmp_path, a):
    anim = animate_power_iteration(a, power_iteration(a, tol=1e-4, max_iter=10))
    assert isinstance(anim, FuncAnimation)
    out = tmp_path / "power.gif"
    anim.save(out, writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 0


def test_animate_power_iteration_requires_iterate_history():
    a = np.diag([2.0, 1.0])
    with pytest.raises(ValueError, match="iterate history"):
        animate_power_iteration(a, eigen_symmetric(a))

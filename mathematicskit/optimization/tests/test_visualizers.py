"""Smoke tests for mathematicskit.optimization.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.optimization.systems.gradient_descent import GradientDescent
from mathematicskit.optimization.systems.newton_quasi_newton import BFGS
from mathematicskit.optimization.utils.comparison import compare_optimizers
from mathematicskit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad
from mathematicskit.optimization.visualizers.plots import plot_contour_path, plot_convergence_comparison


def test_plot_contour_path_returns_axes():
    result = GradientDescent(alpha=0.1, tol=1e-6).minimize(quadratic_bowl, quadratic_bowl_grad, np.array([1.5, 1.0]))
    ax = plot_contour_path(quadratic_bowl, result, n_grid=20)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_convergence_comparison_returns_axes():
    optimizers = {"gd": GradientDescent(alpha=0.05, tol=1e-6), "bfgs": BFGS(tol=1e-6)}
    results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, np.array([3.0, 3.0]))
    ax = plot_convergence_comparison(results, quadratic_bowl, f_star=0.0)
    assert isinstance(ax, matplotlib.axes.Axes)

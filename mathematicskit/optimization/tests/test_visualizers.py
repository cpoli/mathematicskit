"""Smoke tests for mathematicskit.optimization.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mathematicskit.optimization.systems.gradient_descent import GradientDescent
from mathematicskit.optimization.systems.newton_quasi_newton import BFGS
from mathematicskit.optimization.utils.comparison import compare_optimizers
from mathematicskit.optimization.utils.test_functions import quadratic_bowl, quadratic_bowl_grad
from mathematicskit.optimization.visualizers.plots import animate_optimizer_paths, plot_contour_path, plot_convergence_comparison


def test_plot_contour_path_returns_axes():
    result = GradientDescent(alpha=0.1, tol=1e-6).minimize(quadratic_bowl, quadratic_bowl_grad, np.array([1.5, 1.0]))
    ax = plot_contour_path(quadratic_bowl, result, n_grid=20)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_convergence_comparison_returns_axes():
    optimizers = {"gd": GradientDescent(alpha=0.05, tol=1e-6), "bfgs": BFGS(tol=1e-6)}
    results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, np.array([3.0, 3.0]))
    ax = plot_convergence_comparison(results, quadratic_bowl, f_star=0.0)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_animate_optimizer_paths_saves(tmp_path):
    optimizers = {"gd": GradientDescent(alpha=0.05, tol=1e-6), "bfgs": BFGS(tol=1e-6)}
    results = compare_optimizers(optimizers, quadratic_bowl, quadratic_bowl_grad, np.array([2.0, 1.0]))
    anim = animate_optimizer_paths(quadratic_bowl, results, x_range=(-3, 3), y_range=(-3, 3), n_grid=20, n_frames=8)
    assert isinstance(anim, FuncAnimation)
    out = tmp_path / "paths.gif"
    anim.save(out, writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 0

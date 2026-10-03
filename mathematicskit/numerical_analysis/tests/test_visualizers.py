"""Smoke tests for mathematicskit.numerical_analysis.visualizers: right return
type, no exceptions."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mathematicskit.numerical_analysis.systems.chebyshev import chebyshev_nodes, runge_phenomenon_errors
from mathematicskit.numerical_analysis.systems.root_finding import Bisection, NewtonRaphson
from mathematicskit.numerical_analysis.systems.splines import CubicSpline
from mathematicskit.numerical_analysis.visualizers.plots import animate_root_finding, plot_convergence_history, plot_interpolant, plot_runge_phenomenon


def test_plot_convergence_history_returns_axes():
    result = Bisection(lambda x: x**2 - 2.0, 0.0, 2.0).solve()
    ax = plot_convergence_history(result, root_exact=np.sqrt(2.0))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_interpolant_returns_axes():
    spline = CubicSpline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
    ax = plot_interpolant(spline)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_runge_phenomenon_returns_axes():
    degrees = [5, 10, 15]
    equal_err, cheb_err = runge_phenomenon_errors(degrees)
    ax = plot_runge_phenomenon(degrees, equal_err, cheb_err)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_chebyshev_nodes_smoke():
    nodes = chebyshev_nodes(9)
    assert nodes.shape == (9,)


def test_animate_root_finding_saves(tmp_path):
    f = lambda x: x**2 - 2.0
    results = [Bisection(f, 0.0, 2.0, tol=1e-3).solve(), NewtonRaphson(f, lambda x: 2.0 * x, x0=2.0).solve()]
    anim = animate_root_finding(f, results, x_range=(0.0, 2.2))
    assert isinstance(anim, FuncAnimation)
    out = tmp_path / "roots.gif"
    anim.save(out, writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 0

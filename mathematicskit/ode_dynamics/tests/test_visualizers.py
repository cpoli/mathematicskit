"""Smoke tests for mathematicskit.ode_dynamics.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.ode_dynamics.systems.logistic_map import bifurcation_diagram
from mathematicskit.ode_dynamics.systems.phase_portrait import Linear2D, vector_field_grid
from mathematicskit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section
from mathematicskit.ode_dynamics.visualizers.plots import plot_bifurcation_diagram, plot_phase_portrait, plot_poincare_points, plot_vector_field


def test_plot_phase_portrait_returns_axes():
    system = Linear2D([1.0, 1.0], A=[[-1.0, 0.0], [0.0, -1.0]])
    result = system.integrate((0.0, 5.0), dt=1e-3, method="rk4")
    ax = plot_phase_portrait(result)
    assert isinstance(ax, matplotlib.axes.Axes)
    ax2 = plot_phase_portrait([result, result])
    assert isinstance(ax2, matplotlib.axes.Axes)


def test_plot_vector_field_returns_axes():
    X, Y, U, V = vector_field_grid(lambda x, y: (y, -x), (-1, 1), (-1, 1), n=5)
    ax = plot_vector_field(X, Y, U, V)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_bifurcation_diagram_returns_axes():
    r_plot, x_plot = bifurcation_diagram(np.linspace(2.5, 4.0, 5), n_transient=50, n_keep=10)
    ax = plot_bifurcation_diagram(r_plot, x_plot)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_poincare_points_returns_axes():
    system = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2)
    xs, ys = stroboscopic_poincare_section(system, n_periods=4, n_transient_periods=2, dt=1e-2)
    ax = plot_poincare_points(xs, ys)
    assert isinstance(ax, matplotlib.axes.Axes)

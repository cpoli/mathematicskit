"""Smoke tests for mathematicskit.pde.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.pde import HeatEquation1D, HeatEquation2D, solve_poisson_2d
from mathematicskit.pde.visualizers import plot_amplification_factors, plot_field_2d, plot_snapshots, plot_spacetime


def _heat_1d():
    return HeatEquation1D(lambda x: np.sin(np.pi * x), n=11).solve_theta(0.1, dt=0.01)


def test_plot_snapshots_returns_axes():
    ax = plot_snapshots(_heat_1d(), n_snapshots=3)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) == 3


def test_plot_spacetime_returns_axes():
    assert isinstance(plot_spacetime(_heat_1d()), matplotlib.axes.Axes)


def test_plot_field_2d_handles_elliptic_and_time_dependent_solutions():
    elliptic = solve_poisson_2d(1.0, n=(9, 9))
    assert isinstance(plot_field_2d(elliptic), matplotlib.axes.Axes)
    transient = HeatEquation2D(lambda X, Y: X * Y, n=(9, 9)).solve_theta(0.01, dt=1e-3)
    assert isinstance(plot_field_2d(transient, time_index=0), matplotlib.axes.Axes)


def test_plot_amplification_factors_returns_axes():
    ax = plot_amplification_factors(["upwind", "lax_wendroff"], 0.8)
    assert isinstance(ax, matplotlib.axes.Axes)
    assert len(ax.lines) == 3  # two schemes plus the |G| = 1 line

"""Smoke tests for mathematicskit.pde.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mathematicskit.pde import HeatEquation1D, HeatEquation2D, solve_poisson_2d
from mathematicskit.pde.visualizers import animate_solution, plot_amplification_factors, plot_field_2d, plot_snapshots, plot_spacetime


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


def test_animate_solution_1d_with_exact_saves(tmp_path):
    sol = _heat_1d()
    anim = animate_solution(sol, exact=lambda x, t: np.sin(np.pi * x) * np.exp(-(np.pi**2) * t), max_frames=5)
    assert isinstance(anim, FuncAnimation)
    out = tmp_path / "heat.gif"
    anim.save(out, writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 0


def test_animate_solution_2d_with_explicit_frames_saves(tmp_path):
    sol = HeatEquation2D(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y), n=(9, 9)).solve(0.01, dt=5e-4)
    anim = animate_solution(sol, frames=[0, 1, 3, len(sol.t) - 1])
    out = tmp_path / "heat2d.gif"
    anim.save(out, writer=PillowWriter(fps=5), dpi=40)
    assert out.stat().st_size > 0

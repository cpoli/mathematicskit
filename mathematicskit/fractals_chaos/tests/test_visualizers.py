"""Smoke tests for mathematicskit.fractals_chaos.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.fractals_chaos.systems.box_counting import box_counting_dimension
from mathematicskit.fractals_chaos.systems.cellular_automata import ElementaryCA
from mathematicskit.fractals_chaos.systems.ifs import SierpinskiTriangle
from mathematicskit.fractals_chaos.systems.mandelbrot_julia import mandelbrot_set
from mathematicskit.fractals_chaos.visualizers.plots import plot_box_counting, plot_ca_spacetime, plot_escape_time, plot_ifs_points


def test_plot_escape_time_returns_axes():
    result = mandelbrot_set(resolution=15, max_iter=30)
    ax = plot_escape_time(result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_box_counting_returns_axes():
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, size=(2000, 2))
    result = box_counting_dimension(points)
    ax = plot_box_counting(result)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_ifs_points_returns_axes():
    points = SierpinskiTriangle().generate(500, seed=0)
    ax = plot_ifs_points(points)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_ca_spacetime_returns_axes():
    ca = ElementaryCA(rule=30, width=31)
    history = ca.run(20)
    ax = plot_ca_spacetime(history)
    assert isinstance(ax, matplotlib.axes.Axes)

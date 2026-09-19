"""Smoke tests for mathkit.geometry.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathkit.geometry.systems.convex_hull import convex_hull
from mathkit.geometry.systems.curves import frenet_serret_frame
from mathkit.geometry.systems.triangulation import delaunay_triangulation
from mathkit.geometry.utils.curves_library import circle
from mathkit.geometry.visualizers.plots import plot_convex_hull, plot_curve_frame, plot_triangulation, plot_voronoi


def test_plot_convex_hull_returns_axes():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    ax = plot_convex_hull(convex_hull(points))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_triangulation_returns_axes():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ax = plot_triangulation(delaunay_triangulation(points))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_voronoi_returns_axes():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    ax = plot_voronoi(points)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_curve_frame_returns_axes():
    result = frenet_serret_frame(circle(1.0), np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False))
    ax = plot_curve_frame(result)
    assert isinstance(ax, matplotlib.axes.Axes)

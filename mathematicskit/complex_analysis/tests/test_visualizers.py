"""Smoke tests for mathematicskit.complex_analysis.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.complex_analysis.systems.conformal_maps import map_grid
from mathematicskit.complex_analysis.systems.contours import polygon_contour
from mathematicskit.complex_analysis.systems.domain_coloring import domain_coloring
from mathematicskit.complex_analysis.visualizers.plots import plot_contour, plot_domain_coloring, plot_mapped_grid


def test_plot_domain_coloring_returns_axes():
    ax = plot_domain_coloring(domain_coloring(np.sin, resolution=20), title="sin z")
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_contour_returns_axes():
    ax = plot_contour(polygon_contour([0, 1, 1j]), marked_points=[0.2 + 0.2j])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_mapped_grid_returns_two_axes():
    axes = plot_mapped_grid(map_grid(np.exp, (-1, 1), (0, 3), n_lines=3, n_points=10))
    assert axes.shape == (2,)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)

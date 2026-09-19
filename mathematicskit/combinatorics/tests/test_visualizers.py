"""Smoke tests for mathematicskit.combinatorics.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathematicskit.combinatorics.core.base import YoungDiagram
from mathematicskit.combinatorics.systems.pascals_triangle import pascals_triangle
from mathematicskit.combinatorics.visualizers.plots import plot_partition_counts, plot_pascals_triangle, plot_young_diagram


def test_plot_pascals_triangle_returns_axes():
    ax = plot_pascals_triangle(pascals_triangle(8))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_young_diagram_returns_axes():
    ax = plot_young_diagram(YoungDiagram([4, 2, 1]))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_partition_counts_returns_axes():
    ax = plot_partition_counts(20)
    assert isinstance(ax, matplotlib.axes.Axes)

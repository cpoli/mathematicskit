"""Smoke tests for mathkit.combinatorics.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathkit.combinatorics.core.base import YoungDiagram
from mathkit.combinatorics.systems.pascals_triangle import pascals_triangle
from mathkit.combinatorics.visualizers.plots import plot_partition_counts, plot_pascals_triangle, plot_young_diagram


def test_plot_pascals_triangle_returns_axes():
    ax = plot_pascals_triangle(pascals_triangle(8))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_young_diagram_returns_axes():
    ax = plot_young_diagram(YoungDiagram([4, 2, 1]))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_partition_counts_returns_axes():
    ax = plot_partition_counts(20)
    assert isinstance(ax, matplotlib.axes.Axes)

"""Smoke tests for mathkit.abstract_algebra.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathkit.abstract_algebra.systems.groups import CyclicGroup
from mathkit.abstract_algebra.visualizers.plots import plot_cayley_table


def test_plot_cayley_table_returns_axes():
    ax = plot_cayley_table(CyclicGroup(6))
    assert isinstance(ax, matplotlib.axes.Axes)

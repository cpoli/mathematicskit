"""Smoke tests for mathkit.graph_theory.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes

from mathkit.graph_theory.systems.spanning_tree import kruskal_mst
from mathkit.graph_theory.systems.spectral import spectral_analysis
from mathkit.graph_theory.utils.generators import cycle_graph
from mathkit.graph_theory.visualizers.plots import circular_layout, plot_graph, plot_spectral_bipartition


def test_circular_layout_shape():
    positions = circular_layout(6)
    assert positions.shape == (6, 2)


def test_plot_graph_returns_axes():
    g = cycle_graph(6)
    ax = plot_graph(g)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_graph_with_highlighted_mst_edges():
    g = cycle_graph(6)
    mst = kruskal_mst(g)
    ax = plot_graph(g, highlight_edges=[(u, v) for u, v, _w in mst.edges])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_spectral_bipartition_returns_axes():
    g = cycle_graph(8)
    result = spectral_analysis(g)
    ax = plot_spectral_bipartition(g, result)
    assert isinstance(ax, matplotlib.axes.Axes)

"""Smoke tests for mathematicskit.topology.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mathematicskit.topology import circle, mapper_graph, persistent_homology, simplex, torus, vietoris_rips_filtration
from mathematicskit.topology.visualizers import plot_barcode, plot_complex, plot_mapper_graph, plot_persistence_diagram


def test_plot_complex_in_2d_and_3d():
    assert isinstance(plot_complex(simplex(2).skeleton(2), coordinates=[[0, 0], [1, 0], [0, 1]]), matplotlib.axes.Axes)
    assert isinstance(plot_complex(circle(5)), matplotlib.axes.Axes)
    T = torus()
    assert isinstance(plot_complex(T, face_values=np.arange(len(T.simplices(2)))), matplotlib.axes.Axes)
    with pytest.raises(ValueError):
        plot_complex(simplex(3), coordinates=np.zeros((4, 4)))
    plt.close("all")


def test_persistence_plots_return_axes():
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    diagram = persistent_homology(vietoris_rips_filtration(np.column_stack([np.cos(t), np.sin(t)])), max_dim=1)
    assert isinstance(plot_persistence_diagram(diagram), matplotlib.axes.Axes)
    assert isinstance(plot_barcode(diagram), matplotlib.axes.Axes)
    plt.close("all")


def test_plot_mapper_graph_returns_axes():
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    X = np.column_stack([np.cos(t), np.sin(t)])
    assert isinstance(plot_mapper_graph(mapper_graph(X, X[:, 0], eps=0.2, n_intervals=6), X), matplotlib.axes.Axes)
    plt.close("all")

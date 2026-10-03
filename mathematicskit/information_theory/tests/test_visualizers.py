"""Smoke tests for mathematicskit.information_theory.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.information_theory import huffman_code
from mathematicskit.information_theory.visualizers import plot_code_tree, plot_information_diagram


def test_plot_code_tree_returns_axes():
    ax = plot_code_tree(huffman_code({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_information_diagram_returns_axes():
    ax = plot_information_diagram(0.5 * np.array([[0.9, 0.1], [0.1, 0.9]]))
    assert isinstance(ax, matplotlib.axes.Axes)

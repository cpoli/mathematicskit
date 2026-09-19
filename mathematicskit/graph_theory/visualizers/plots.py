"""Plotting helpers for mathematicskit.graph_theory: a dependency-free circular
graph layout, with highlighting for spanning trees, colorings, and
spectral bipartitions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["circular_layout", "plot_graph", "plot_spectral_bipartition"]


def circular_layout(n_vertices: int) -> np.ndarray:
    """Place `n_vertices` points evenly around a unit circle.

    A simple, dependency-free layout (no ``networkx``) good enough for
    the small graphs in this domain's examples/tests.

    Parameters
    ----------
    n_vertices : int

    Returns
    -------
    ndarray, shape (n_vertices, 2)
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def plot_graph(graph, ax=None, positions=None, highlight_edges=None, vertex_colors=None):
    """Draw a graph with a circular layout.

    Parameters
    ----------
    graph : Graph
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    positions : ndarray, shape (n, 2), optional
        Defaults to :func:`circular_layout`.
    highlight_edges : list of (int, int), optional
        Edges to draw in a highlight color (e.g. a spanning tree's edges).
    vertex_colors : dict, optional
        ``{vertex: color_index}`` (e.g. from
        :func:`~mathematicskit.graph_theory.systems.coloring.greedy_coloring`).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    positions = circular_layout(graph.n_vertices) if positions is None else positions
    highlight_edges = set() if highlight_edges is None else {tuple(sorted(e)) for e in highlight_edges}

    for u, v, _w in graph.edges():
        is_highlighted = tuple(sorted((u, v))) in highlight_edges
        ax.plot(
            [positions[u, 0], positions[v, 0]],
            [positions[u, 1], positions[v, 1]],
            color="firebrick" if is_highlighted else "gray",
            linewidth=2.0 if is_highlighted else 1.0,
            zorder=1,
        )

    if vertex_colors:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(vertex_colors[v] % 10) for v in range(graph.n_vertices)]
    else:
        colors = "steelblue"
    ax.scatter(positions[:, 0], positions[:, 1], s=300, c=colors, zorder=2, edgecolors="black")
    for v in range(graph.n_vertices):
        ax.annotate(str(v), positions[v], ha="center", va="center", zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_spectral_bipartition(graph, spectral_result, ax=None):
    """Draw a graph colored by its spectral (Fiedler-vector) bipartition.

    Parameters
    ----------
    graph : Graph
    spectral_result : SpectralResult
        From :func:`~mathematicskit.graph_theory.systems.spectral.spectral_analysis`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    vertex_colors = {v: int(spectral_result.bipartition[v]) for v in range(graph.n_vertices)}
    ax = plot_graph(graph, ax=ax, vertex_colors=vertex_colors)
    ax.set_title(f"Spectral bipartition (algebraic connectivity = {spectral_result.algebraic_connectivity:.3f})")
    return ax

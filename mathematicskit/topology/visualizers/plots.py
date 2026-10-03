"""Plotting helpers for mathematicskit.topology: simplicial complexes in 2D or 3D,
persistence diagrams and barcodes, and Mapper graphs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

__all__ = ["plot_complex", "plot_persistence_diagram", "plot_barcode", "plot_mapper_graph"]


def plot_complex(K, ax=None, coordinates=None, face_values=None, cmap="viridis", face_color="tab:blue", alpha=0.35, vertex_size=10.0):
    """Draw the triangles, edges and vertices of a complex at its vertex coordinates (2D or 3D).

    Parameters
    ----------
    K : SimplicialComplex
    ax : matplotlib.axes.Axes, optional
        A 3D axes for 3D coordinates; a new figure is created if omitted.
    coordinates : array_like, shape (n_vertices, 2 or 3), optional
        Overrides ``K.coordinates``.
    face_values : array_like, optional
        One value per triangle (in ``K.simplices(2)`` order), colouring the faces through ``cmap``.
    cmap : str
    face_color : color
        Face colour when ``face_values`` is omitted.
    alpha : float
        Face opacity.
    vertex_size : float
        Marker area of the vertices; 0 hides them.

    Returns
    -------
    matplotlib.axes.Axes
    """
    coords = np.asarray(K.coordinates if coordinates is None else coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("plot_complex needs 2D or 3D vertex coordinates")
    three_d = coords.shape[1] == 3
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d" if three_d else None)
    triangles = [coords[list(t)] for t in K.simplices(2)]
    edges = [coords[list(e)] for e in K.simplices(1)]
    if three_d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

        faces = Poly3DCollection(triangles, alpha=alpha, edgecolor="none")
        lines = Line3DCollection(edges, colors="0.25", linewidths=0.6)
    else:
        faces = PolyCollection(triangles, alpha=alpha, edgecolor="none")
        lines = LineCollection(edges, colors="0.25", linewidths=0.8)
    if face_values is not None:
        faces.set_array(np.asarray(face_values, dtype=float))
        faces.set_cmap(cmap)
    else:
        faces.set_facecolor(face_color)
    if triangles:
        ax.add_collection(faces)
    ax.add_collection(lines)
    vertices = coords[K.vertices]
    if vertex_size:
        ax.scatter(*vertices.T, s=vertex_size, color="k", zorder=3)
    if three_d:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        center, half = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.autoscale_view()
        ax.set_aspect("equal")
    return ax


def _finite_ceiling(diagram, dims) -> float:
    finite = np.concatenate([diagram.diagram(k).ravel() for k in dims] + [np.zeros(1)])
    finite = finite[np.isfinite(finite)]
    return float(finite.max()) * 1.1 if finite.max() > 0 else 1.0


def plot_persistence_diagram(diagram, ax=None, dims=None):
    """Plot (birth, death) points per homology dimension, with the diagonal; infinite deaths sit on a dashed line at the top.

    Parameters
    ----------
    diagram : PersistenceDiagram
    ax : matplotlib.axes.Axes, optional
    dims : iterable of int, optional
        Dimensions to show; all by default.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    dims = list(range(diagram.max_dimension + 1)) if dims is None else list(dims)
    top = _finite_ceiling(diagram, dims)
    ax.plot([0, top], [0, top], color="0.5", lw=1)
    ax.axhline(top, color="0.5", ls="--", lw=1)
    for k in dims:
        d = diagram.diagram(k)
        deaths = np.where(np.isinf(d[:, 1]), top, d[:, 1])
        ax.scatter(d[:, 0], deaths, s=20, label=f"$H_{k}$", zorder=3)
    ax.text(0.02 * top, top, r"$\infty$", va="bottom")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_xlim(-0.03 * top, top * 1.03)
    ax.set_ylim(-0.03 * top, top * 1.08)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax


def plot_barcode(diagram, ax=None, dims=None):
    """Draw each persistence interval as a horizontal bar, grouped by dimension; infinite bars end in an arrow.

    Parameters
    ----------
    diagram : PersistenceDiagram
    ax : matplotlib.axes.Axes, optional
    dims : iterable of int, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    dims = list(range(diagram.max_dimension + 1)) if dims is None else list(dims)
    top = _finite_ceiling(diagram, dims)
    row = 0
    for c, k in enumerate(dims):
        d = diagram.diagram(k)
        d = d[np.argsort(d[:, 0], kind="stable")]
        color = f"C{c}"
        for b, e in d:
            ax.plot([b, min(e, top)], [row, row], color=color, lw=2, solid_capstyle="butt")
            if np.isinf(e):
                ax.annotate("", (top, row), (top * 0.97, row), arrowprops={"arrowstyle": "->", "color": color})
            row += 1
        ax.plot([], [], color=color, lw=2, label=f"$H_{k}$")
    ax.set_xlim(0, top * 1.02)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("filtration value")
    ax.legend(loc="lower right")
    return ax


def plot_mapper_graph(result, points, ax=None, cmap="viridis"):
    """Draw a Mapper graph with each node at the mean of its cluster's points, coloured by its lens value and sized by its cluster.

    Parameters
    ----------
    result : MapperResult
    points : array_like, shape (n, d)
        The point cloud; the first two coordinates place the nodes.
    ax : matplotlib.axes.Axes, optional
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    pts = np.asarray(points, dtype=float)[:, :2]
    pos = np.array([pts[m].mean(axis=0) for m in result.nodes])
    ax.add_collection(LineCollection([pos[list(e)] for e in result.edges], colors="0.4", linewidths=1.5, zorder=1))
    sizes = 20 + 200 * np.array([len(m) for m in result.nodes]) / max(len(m) for m in result.nodes)
    ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=result.node_values, cmap=cmap, edgecolors="k", zorder=2)
    ax.set_aspect("equal")
    ax.autoscale_view()
    return ax

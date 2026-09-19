"""Plotting helpers for mathematicskit.geometry: convex hulls, triangulations,
Voronoi diagrams, and curve frames."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

__all__ = ["plot_convex_hull", "plot_triangulation", "plot_voronoi", "plot_curve_frame"]


def plot_convex_hull(result, ax=None):
    """Plot a 2D point set with its convex hull outlined.

    Parameters
    ----------
    result : ConvexHullResult
        From :func:`~mathematicskit.geometry.systems.convex_hull.convex_hull` or
        :func:`~mathematicskit.geometry.systems.convex_hull.graham_scan`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    points = result.points
    ax.scatter(points[:, 0], points[:, 1], color="steelblue", zorder=2)
    hull_points = points[result.vertices]
    closed = np.vstack([hull_points, hull_points[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="firebrick", lw=2, zorder=1)
    ax.set_title(f"Convex hull ({result.method})")
    ax.set_aspect("equal")
    return ax


def plot_triangulation(result, ax=None):
    """Plot a Delaunay triangulation.

    Parameters
    ----------
    result : TriangulationResult
        From :func:`~mathematicskit.geometry.systems.triangulation.delaunay_triangulation`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    points = result.points
    ax.triplot(points[:, 0], points[:, 1], result.simplices, color="gray")
    ax.scatter(points[:, 0], points[:, 1], color="steelblue", zorder=2)
    ax.set_title("Delaunay triangulation")
    ax.set_aspect("equal")
    return ax


def plot_voronoi(points: np.ndarray, ax=None):
    """Plot a Voronoi diagram, via :func:`scipy.spatial.voronoi_plot_2d`.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax)
    ax.set_title("Voronoi diagram")
    ax.set_aspect("equal")
    return ax


def plot_curve_frame(result, ax=None, frame_stride: int = 20, scale: float = 0.3):
    """Plot a parametric curve with its tangent/normal frame vectors sampled along it.

    Parameters
    ----------
    result : CurveFrameResult
        From :func:`~mathematicskit.geometry.systems.curves.frenet_serret_frame`
        on a 2D curve.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    frame_stride : int
        Plot a tangent/normal pair every `frame_stride` samples.
    scale : float
        Length of the plotted frame vectors.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.position[:, 0], result.position[:, 1], color="steelblue", lw=2)
    for i in range(0, len(result.t), frame_stride):
        p = result.position[i]
        ax.arrow(p[0], p[1], scale * result.tangent[i, 0], scale * result.tangent[i, 1], color="firebrick", head_width=0.05, length_includes_head=True)
        ax.arrow(p[0], p[1], scale * result.normal[i, 0], scale * result.normal[i, 1], color="forestgreen", head_width=0.05, length_includes_head=True)
    ax.set_title("Frenet-Serret frame (red=tangent, green=normal)")
    ax.set_aspect("equal")
    return ax

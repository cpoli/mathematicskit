"""Plotting helpers for mathematicskit.complex_analysis: domain-coloring images,
contours with marked points, and conformally mapped grids."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_domain_coloring", "plot_contour", "plot_mapped_grid"]


def plot_domain_coloring(result, ax=None, title: str | None = None):
    """Show a :class:`~mathematicskit.complex_analysis.core.base.DomainColoringResult` as an image.

    Parameters
    ----------
    result : DomainColoringResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    z = result.z
    extent = (z.real.min(), z.real.max(), z.imag.min(), z.imag.max())
    ax.imshow(result.rgb, origin="lower", extent=extent, interpolation="bilinear")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_contour(contour, marked_points=(), ax=None, n_points: int = 400):
    """Draw a contour with direction arrows, and optionally mark points (poles, zeros).

    Parameters
    ----------
    contour : Contour
    marked_points : sequence of complex
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    n_points : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    z = contour.points(n_points)
    ax.plot(z.real, z.imag, "C0")
    for k in np.linspace(0, n_points - 2, 5, dtype=int)[1:]:
        ax.annotate("", xy=(z[k + 1].real, z[k + 1].imag), xytext=(z[k].real, z[k].imag), arrowprops={"arrowstyle": "->", "color": "C0"})
    points = np.asarray(marked_points, dtype=complex)
    if points.size:
        ax.plot(points.real, points.imag, "rx")
    ax.set_aspect("equal")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    return ax


def plot_mapped_grid(grid, axes=None):
    """Side-by-side plots of a grid in the z-plane and its image in the w-plane.

    Parameters
    ----------
    grid : MappedGrid
    axes : sequence of two matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, h, v, label in ((axes[0], grid.horizontal, grid.vertical, "z"), (axes[1], grid.horizontal_image, grid.vertical_image, "w")):
        for line in h:
            ax.plot(line.real, line.imag, "C0", lw=0.8)
        for line in v:
            ax.plot(line.real, line.imag, "C1", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Re {label}")
        ax.set_ylabel(f"Im {label}")
    return np.asarray(axes)

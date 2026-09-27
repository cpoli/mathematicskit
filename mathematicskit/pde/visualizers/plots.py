"""Plotting helpers for mathematicskit.pde: solution snapshots, space-time
diagrams, 2D fields, and von Neumann amplification factors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.pde.systems.stability import amplification_factor

__all__ = ["plot_snapshots", "plot_spacetime", "plot_field_2d", "plot_amplification_factors"]


def plot_snapshots(solution, n_snapshots: int = 5, ax=None, cmap: str = "viridis"):
    """Overlay a 1D solution at `n_snapshots` evenly spaced saved times.

    Parameters
    ----------
    solution : PDESolution
        A 1D solve (``solution.u`` of shape ``(n_t, n_x)``).
    n_snapshots : int
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    cmap : str
        Colormap used to color curves from early to late.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    indices = np.unique(np.linspace(0, len(solution.t) - 1, n_snapshots).round().astype(int))
    colors = plt.get_cmap(cmap)(np.linspace(0.0, 0.9, len(indices)))
    for color, k in zip(colors, indices, strict=False):
        ax.plot(solution.x, solution.u[k], color=color, label=f"t = {solution.t[k]:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=8)
    return ax


def plot_spacetime(solution, ax=None, cmap: str = "viridis"):
    """Space-time heatmap ``u(x, t)`` of a 1D solve, time increasing upward.

    Parameters
    ----------
    solution : PDESolution
    ax : matplotlib.axes.Axes, optional
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(solution.x, solution.t, solution.u, shading="auto", cmap=cmap)
    ax.figure.colorbar(mesh, ax=ax, label="u")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return ax


def plot_field_2d(solution, time_index: int = -1, ax=None, levels: int = 20, cmap: str = "viridis"):
    """Filled contour plot of a 2D field.

    Parameters
    ----------
    solution : EllipticSolution or PDESolution
        A 2D solve; for a time-dependent one, `time_index` picks the snapshot.
    time_index : int
    ax : matplotlib.axes.Axes, optional
    levels : int
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    u = solution.u[time_index] if solution.u.ndim == 3 else solution.u
    X, Y = np.meshgrid(solution.x, solution.y, indexing="ij")
    filled = ax.contourf(X, Y, u, levels=levels, cmap=cmap)
    ax.figure.colorbar(filled, ax=ax, label="u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return ax


def plot_amplification_factors(schemes, number: float, ax=None):
    r"""Plot :math:`|G(\xi)|` over :math:`\xi \in [0, \pi]` for several schemes, with the stability line :math:`|G| = 1`.

    Parameters
    ----------
    schemes : iterable of str
        Keys of :data:`~mathematicskit.pde.systems.stability.AMPLIFICATION_SCHEMES`.
    number : float
        Courant or diffusion number shared by all curves.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    xi = np.linspace(0.0, np.pi, 361)
    for scheme in schemes:
        ax.plot(xi, np.abs(amplification_factor(scheme, number, xi)), label=scheme)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$\xi = k\,\Delta x$")
    ax.set_ylabel(r"$|G(\xi)|$")
    ax.legend(fontsize=8)
    return ax

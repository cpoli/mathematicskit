"""Plotting helpers for mathematicskit.pde: solution snapshots, space-time
diagrams, 2D fields, von Neumann amplification factors, and an animation
of a time-dependent solve.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mathematicskit.pde.systems.stability import amplification_factor

__all__ = ["plot_snapshots", "plot_spacetime", "plot_field_2d", "plot_amplification_factors", "animate_solution"]


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


def animate_solution(solution, exact=None, max_frames: int = 80, frames=None, interval: int = 60, cmap: str = "viridis"):
    """Animate a time-dependent solve, one saved time step per frame.

    A 1D solve is drawn as the curve ``u(x)`` on fixed axes, so decay
    (heat equation) or travel and reflection (wave equation) read
    directly; a 2D solve is drawn as an image with a fixed color scale.

    Parameters
    ----------
    solution : PDESolution
        From any :class:`~mathematicskit.pde.core.base.MethodOfLinesPDE`
        solve, e.g. :class:`~mathematicskit.pde.systems.heat.HeatEquation1D`
        or :class:`~mathematicskit.pde.systems.wave.WaveEquation1D`.
    exact : callable, optional
        ``exact(x, t) -> ndarray``, overlaid as a dashed curve (1D only),
        e.g. a wrapper around
        :func:`~mathematicskit.pde.systems.heat.heat_series_solution` or
        :func:`~mathematicskit.pde.systems.wave.dalembert_solution`.
    max_frames : int
        Saved time steps are thinned evenly to at most this many frames.
    frames : array-like of int, optional
        Indices into ``solution.t`` to show instead, e.g. geometrically
        spaced ones to follow fast early decay; overrides `max_frames`.
    interval : int
        Delay between frames, in milliseconds.
    cmap : str
        Colormap of a 2D solve.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import numpy as np
    >>> from mathematicskit.pde import HeatEquation1D
    >>> sol = HeatEquation1D(lambda x: np.sin(np.pi * x), n=21).solve(0.1, method="rk4", dt=1e-3, save_every=10)
    >>> anim = animate_solution(sol, exact=lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi**2 * t), max_frames=5)
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    u = np.asarray(solution.u)
    if frames is None:
        frames = np.linspace(0, len(solution.t) - 1, min(max_frames, len(solution.t))).round()
    frames = np.unique(np.asarray(frames, dtype=int))
    lo, hi = float(np.min(u)), float(np.max(u))
    pad = 0.05 * (hi - lo) or 0.1

    fig, ax = plt.subplots(figsize=(7, 4.5) if u.ndim == 2 else (6, 5))
    if u.ndim == 2:
        (line,) = ax.plot(solution.x, u[0], color="tab:blue", lw=2, label=solution.method or "numerical")
        artists: list = [line]
        if exact is not None:
            (exact_line,) = ax.plot(solution.x, exact(solution.x, solution.t[0]), "k--", lw=1, label="exact")
            artists.append(exact_line)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(solution.x[0], solution.x[-1])
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
    else:
        image = ax.imshow(u[0].T, origin="lower", extent=(solution.x[0], solution.x[-1], solution.y[0], solution.y[-1]), vmin=lo, vmax=hi, cmap=cmap)
        fig.colorbar(image, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        artists = [image]
    title = ax.set_title("")
    fig.tight_layout()

    def update(frame):
        k = frames[frame]
        t = solution.t[k]
        if u.ndim == 2:
            artists[0].set_ydata(u[k])
            if exact is not None:
                artists[1].set_ydata(exact(solution.x, t))
        else:
            artists[0].set_data(u[k].T)
        title.set_text(f"t = {t:.3g}")
        return [*artists, title]

    return FuncAnimation(fig, update, frames=len(frames), interval=interval)

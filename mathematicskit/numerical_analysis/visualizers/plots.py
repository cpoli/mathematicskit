"""Plotting helpers for root-finding convergence, interpolants, and the
Runge-phenomenon comparison, plus an animation of root finders stepping
along the graph of ``f``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["plot_convergence_history", "plot_interpolant", "plot_runge_phenomenon", "animate_root_finding"]


def plot_convergence_history(result, root_exact=None, ax=None, label=None):
    """Semilog plot of ``|x_n - root|`` vs. iteration, for a :class:`~mathematicskit.numerical_analysis.core.base.RootResult`.

    Parameters
    ----------
    result : RootResult
        A solved root-finder result (``result.history``).
    root_exact : float, optional
        Reference root to measure error against; defaults to
        ``result.root`` (the sequence's own final iterate).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional
        Legend label; defaults to ``result.method``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    root = result.root if root_exact is None else root_exact
    errors = np.abs(result.history - root)
    errors = np.where(errors == 0.0, np.nan, errors)  # avoid log(0)
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(np.arange(len(errors)), errors, marker="o", ms=3, label=label or result.method)
    ax.set_xlabel("iteration")
    ax.set_ylabel("|x_n - root|")
    ax.set_title(f"Convergence history ({result.method})")
    return ax


def plot_interpolant(interpolant, x_fine=None, ax=None, show_nodes=True):
    """Plot an interpolant's curve alongside its data nodes.

    Parameters
    ----------
    interpolant : Interpolant
        Any concrete :class:`~mathematicskit.numerical_analysis.core.base.Interpolant`
        (e.g. :class:`~mathematicskit.numerical_analysis.systems.splines.CubicSpline`).
    x_fine : ndarray, optional
        Evaluation grid; defaults to 400 points spanning the nodes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    show_nodes : bool
        Whether to scatter the original data nodes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if x_fine is None:
        x_fine = np.linspace(interpolant.x.min(), interpolant.x.max(), 400)
    y_fine = interpolant.evaluate(x_fine)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x_fine, y_fine, color="steelblue", label="interpolant")
    if show_nodes:
        ax.scatter(interpolant.x, interpolant.y, color="firebrick", zorder=3, label="nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(type(interpolant).__name__)
    ax.legend()
    return ax


def plot_runge_phenomenon(degrees, equal_errors, chebyshev_errors, ax=None):
    """Semilog plot of max interpolation error vs. degree, equally spaced
    vs. Chebyshev nodes (see
    :func:`mathematicskit.numerical_analysis.systems.chebyshev.runge_phenomenon_errors`).

    Parameters
    ----------
    degrees : array-like of int
    equal_errors, chebyshev_errors : array-like of float
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.semilogy(degrees, equal_errors, "o-", color="firebrick", label="equally spaced")
    ax.semilogy(degrees, chebyshev_errors, "o-", color="steelblue", label="Chebyshev")
    ax.set_xlabel("polynomial degree")
    ax.set_ylabel("max |error|")
    ax.set_title("Runge phenomenon: node placement matters")
    ax.legend()
    return ax


def animate_root_finding(f, results, x_range, root_exact=None, n_grid: int = 400, interval: int = 700):
    """Animate one or more root finders stepping along the graph of `f`.

    Each :class:`~mathematicskit.numerical_analysis.core.base.RootResult`
    gets its own panel showing ``f`` and the current iterate. A result
    carrying ``extra["brackets"]`` (from
    :class:`~mathematicskit.numerical_analysis.systems.root_finding.Bisection`)
    shows the bracket being halved; any other method shows the step from
    ``(x_{k-1}, f(x_{k-1}))`` to ``x_k`` on the axis, which for
    :class:`~mathematicskit.numerical_analysis.systems.root_finding.NewtonRaphson`
    is the tangent line. A shared bottom panel grows the semilog error
    ``|x_k - root|`` of every method, frame by frame, so linear and
    quadratic convergence separate as the animation plays.

    Parameters
    ----------
    f : callable
        The function whose root was sought, ``f(x) -> float``.
    results : RootResult or sequence of RootResult
        Solved root-finder results (``result.history``).
    x_range : tuple of float
        ``(x_min, x_max)`` of the plotted graph.
    root_exact : float, optional
        Reference root for the error panel; defaults to the result root
        with the smallest ``|f(root)|``.
    n_grid : int
        Points used to draw the graph of `f`.
    interval : int
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        One frame per iteration of the longest-running method; methods
        that finish early hold their final iterate.

    Examples
    --------
    >>> from mathematicskit.numerical_analysis import Bisection, NewtonRaphson
    >>> f = lambda x: x**2 - 2.0
    >>> results = [Bisection(f, 0.0, 2.0, tol=1e-3).solve(), NewtonRaphson(f, lambda x: 2.0 * x, x0=2.0).solve()]
    >>> anim = animate_root_finding(f, results, x_range=(0.0, 2.2))
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    if root_exact is None:
        root_exact = min((r.root for r in results), key=lambda x: abs(f(x)))
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.array([f(x) for x in xs])

    n_methods = len(results)
    fig = plt.figure(figsize=(4.5 * n_methods + 1.0, 7))
    grid = fig.add_gridspec(2, n_methods, height_ratios=(3, 2))
    ax_err = fig.add_subplot(grid[1, :])
    errors = [np.abs(np.asarray(r.history) - root_exact) for r in results]
    positive = np.concatenate([e[e > 0] for e in errors])
    ax_err.set_yscale("log")
    ax_err.set_xlim(-0.5, max(len(e) for e in errors) - 0.5)
    if positive.size:
        ax_err.set_ylim(positive.min() / 3.0, positive.max() * 3.0)
    ax_err.set_xlabel("iteration k")
    ax_err.set_ylabel("|x_k - root|")

    panels = []
    for col, result in enumerate(results):
        ax = fig.add_subplot(grid[0, col])
        ax.plot(xs, ys, color="0.3", lw=1.5)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_xlim(*x_range)
        ax.set_xlabel("x")
        ax.set_title(result.method)
        (trail,) = ax.plot([], [], "o", color="tab:blue", alpha=0.35, ms=4)
        (step,) = ax.plot([], [], "-", color="tab:red", lw=1.2)
        (drop,) = ax.plot([], [], ":", color="tab:red", lw=1.0)
        (current,) = ax.plot([], [], "o", color="tab:red", ms=7, zorder=3)
        (bracket,) = ax.plot([], [], "-", color="tab:orange", lw=8, alpha=0.4, solid_capstyle="butt")
        (err_line,) = ax_err.plot([], [], "o-", ms=4, label=result.method)
        panels.append((result, np.asarray(result.history), trail, step, drop, current, bracket, err_line))
    ax_err.legend(fontsize=8)
    title = fig.suptitle("")
    fig.tight_layout()

    n_frames = max(len(e) for e in errors)

    def update(frame):
        artists = [title]
        for (result, history, trail, step, drop, current, bracket, err_line), err in zip(panels, errors, strict=True):
            k = min(frame, len(history) - 1)
            x_k = history[k]
            trail.set_data(history[:k], np.zeros(k))
            current.set_data([x_k], [0.0])
            drop.set_data([x_k, x_k], [0.0, f(x_k)])
            if "brackets" in result.extra:
                a, b = result.extra["brackets"][k]
                bracket.set_data([a, b], [0.0, 0.0])
            elif k > 0:
                step.set_data([history[k - 1], x_k], [f(history[k - 1]), 0.0])
            else:
                step.set_data([], [])
            err_line.set_data(np.arange(k + 1), np.where(err[: k + 1] > 0, err[: k + 1], np.nan))
            artists += [trail, step, drop, current, bracket, err_line]
        title.set_text(f"iteration k = {frame}")
        return artists

    return FuncAnimation(fig, update, frames=n_frames, interval=interval)

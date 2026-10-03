"""Plotting helpers for mathematicskit.optimization: 2D contour + iterate-path
plots, convergence-rate comparison plots, and an animation of iterate
paths advancing over the contours.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mathematicskit.optimization.core.base import OptimizeResult

__all__ = ["plot_contour_path", "plot_convergence_comparison", "animate_optimizer_paths"]


def _objective_grid(f: Callable[[np.ndarray], float], x_range, y_range, n_grid: int):
    """Evaluate a 2D objective on an ``n_grid`` x ``n_grid`` mesh, returning ``(X, Y, Z)``."""
    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    return X, Y, Z


def plot_contour_path(f: Callable[[np.ndarray], float], result, ax=None, x_range=(-2.0, 2.0), y_range=(-1.0, 3.0), n_grid: int = 200, label=None):
    """Contour plot of a 2D objective with an optimizer's iterate path overlaid.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float`` (called on 2-vectors).
    result : OptimizeResult
        From any :class:`~mathematicskit.optimization.core.base.UnconstrainedOptimizer`,
        solved on a 2D problem.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    x_range, y_range : tuple of float
        Contour grid extent.
    n_grid : int
        Grid resolution per axis.
    label : str, optional
        Legend label for the path; defaults to ``result.method``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    X, Y, Z = _objective_grid(f, x_range, y_range, n_grid)
    ax.contour(X, Y, Z, levels=30, cmap="Greys", linewidths=0.5)
    path = np.asarray(result.path)
    ax.plot(path[:, 0], path[:, 1], "o-", ms=3, lw=1, label=label or result.method)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimizer iterate path")
    ax.legend()
    return ax


def plot_convergence_comparison(results: dict[str, OptimizeResult], f: Callable[[np.ndarray], float], f_star: float, ax=None):
    """Semilog plot of ``f(x_k) - f*`` vs. iteration, across several methods.

    Parameters
    ----------
    results : dict of str -> OptimizeResult
        E.g. from :func:`~mathematicskit.optimization.utils.comparison.compare_optimizers`.
    f : callable
    f_star : float
        Known (or best available) optimal value.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    for name, result in results.items():
        gap = np.array([f(x) - f_star for x in result.path])
        gap = np.where(gap <= 0.0, np.nan, gap)
        ax.semilogy(np.arange(len(gap)), gap, "o-", ms=3, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f(x_k) - f*")
    ax.set_title("Convergence-rate comparison")
    ax.legend()
    return ax


def animate_optimizer_paths(
    f: Callable[[np.ndarray], float],
    results,
    x_range=(-2.0, 2.0),
    y_range=(-1.0, 3.0),
    n_grid: int = 200,
    levels=30,
    n_frames: int = 60,
    interval: int = 120,
):
    """Animate optimizer iterate paths advancing over the contours of a 2D objective.

    Frames are spaced geometrically in the iteration count (0, 1, 2, ...,
    then ever larger jumps), so a method that needs a handful of steps and
    one that needs thousands share one animation: the title shows how many
    iterations each frame has reached.

    Parameters
    ----------
    f : callable
        Objective ``f(x) -> float`` (called on 2-vectors).
    results : OptimizeResult or dict of str -> OptimizeResult
        From any :class:`~mathematicskit.optimization.core.base.UnconstrainedOptimizer`
        solved on a 2D problem, e.g. the output of
        :func:`~mathematicskit.optimization.utils.comparison.compare_optimizers`.
    x_range, y_range : tuple of float
        Contour grid extent.
    n_grid : int
        Grid resolution per axis.
    levels : int or array-like
        Forwarded to :meth:`~matplotlib.axes.Axes.contour`; pass
        geometrically spaced levels for objectives with a deep narrow
        valley, such as :func:`~mathematicskit.optimization.utils.test_functions.rosenbrock`.
    n_frames : int
        Maximum number of frames.
    interval : int
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> from mathematicskit.optimization import NonlinearConjugateGradient, quadratic_bowl, quadratic_bowl_grad
    >>> result = NonlinearConjugateGradient(tol=1e-8).minimize(quadratic_bowl, quadratic_bowl_grad, [2.0, 1.0])
    >>> anim = animate_optimizer_paths(quadratic_bowl, result, x_range=(-3, 3), y_range=(-3, 3), n_grid=40)
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    if isinstance(results, OptimizeResult):
        results = {results.method: results}
    paths = {name: np.asarray(result.path) for name, result in results.items()}
    longest = max(len(path) for path in paths.values()) - 1
    counts = np.unique(np.concatenate([[0], np.round(np.geomspace(1, max(longest, 1), max(n_frames - 1, 1)))]).astype(int))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    X, Y, Z = _objective_grid(f, x_range, y_range, n_grid)
    ax.contour(X, Y, Z, levels=levels, colors="0.65", linewidths=0.6)
    lines = {}
    for name, path in paths.items():
        (line,) = ax.plot(path[:1, 0], path[:1, 1], "o-", ms=3, lw=1, label=name)
        (head,) = ax.plot(path[:1, 0], path[:1, 1], "o", ms=8, color=line.get_color(), markeredgecolor="k")
        lines[name] = (line, head)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    title = ax.set_title("")
    fig.tight_layout()

    def update(frame):
        k = counts[frame]
        artists = [title]
        for name, path in paths.items():
            line, head = lines[name]
            stop = min(k, len(path) - 1)
            line.set_data(path[: stop + 1, 0], path[: stop + 1, 1])
            head.set_data(path[stop : stop + 1, 0], path[stop : stop + 1, 1])
            artists += [line, head]
        title.set_text(f"iteration {k}")
        return artists

    return FuncAnimation(fig, update, frames=len(counts), interval=interval)

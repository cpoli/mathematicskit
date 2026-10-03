"""Plotting helpers for mathematicskit.ode_dynamics: phase portraits, bifurcation
diagrams, Poincare sections, integrator stability regions, and an
animation of the logistic map's cobweb diagram as its parameter grows."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

from mathematicskit.integrators.stability import is_absolutely_stable
from mathematicskit.ode_dynamics.systems.logistic_map import LogisticMap, bifurcation_diagram

__all__ = ["plot_phase_portrait", "plot_vector_field", "plot_bifurcation_diagram", "plot_poincare_points", "plot_stability_regions", "animate_logistic_cobweb"]


def plot_phase_portrait(trajectories, ax=None, **kwargs):
    """Plot one or more ``(x, y)`` trajectories in phase space.

    Parameters
    ----------
    trajectories : OdeTrajectory or list of OdeTrajectory
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if not isinstance(trajectories, (list, tuple)):
        trajectories = [trajectories]
    for traj in trajectories:
        ax.plot(traj.y[:, 0], traj.y[:, 1], **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase portrait")
    return ax


def plot_vector_field(X, Y, U, V, ax=None):
    """Quiver plot of a sampled vector field.

    Parameters
    ----------
    X, Y, U, V : ndarray
        From :func:`mathematicskit.ode_dynamics.systems.phase_portrait.vector_field_grid`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    speed = np.sqrt(U**2 + V**2)
    speed_safe = np.where(speed == 0.0, 1.0, speed)
    ax.quiver(X, Y, U / speed_safe, V / speed_safe, speed, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Vector field")
    return ax


def plot_bifurcation_diagram(r_plot, x_plot, ax=None):
    """Scatter plot of a 1D map's bifurcation diagram.

    Parameters
    ----------
    r_plot, x_plot : ndarray
        From :func:`mathematicskit.ode_dynamics.systems.logistic_map.bifurcation_diagram`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(r_plot, x_plot, ",", color="black", alpha=0.5)
    ax.set_xlabel("r")
    ax.set_ylabel("x (after transient)")
    ax.set_title("Bifurcation diagram")
    return ax


def plot_poincare_points(xs, ys, ax=None):
    """Scatter plot of stroboscopic Poincare-section points.

    Parameters
    ----------
    xs, ys : ndarray
        From :func:`mathematicskit.ode_dynamics.systems.poincare.stroboscopic_poincare_section`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(xs, ys, s=6, color="firebrick")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Poincare section")
    return ax


def plot_stability_regions(methods, ax=None, re_range=(-4.0, 1.5), im_range=(-3.5, 3.5), n_grid: int = 400, labels=None):
    r"""Shade the absolute-stability regions of integrators in the :math:`z = \lambda h` plane.

    Parameters
    ----------
    methods : str or list of str
        Names accepted by :func:`mathematicskit.integrators.is_absolutely_stable`
        (e.g. ``"euler"``, ``"rk4"``, ``"adams_bashforth3"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    re_range, im_range : tuple of float
        Extent of the sampled grid.
    n_grid : int
        Grid points per axis.
    labels : list of str, optional
        Legend labels; defaults to the method names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if isinstance(methods, str):
        methods = [methods]
    labels = list(methods) if labels is None else list(labels)
    x, y = np.meshgrid(np.linspace(re_range[0], re_range[1], n_grid), np.linspace(im_range[0], im_range[1], n_grid))
    handles = []
    for k, (method, label) in enumerate(zip(methods, labels, strict=True)):
        color = f"C{k}"
        stable = is_absolutely_stable(method, x + 1j * y).astype(float)
        ax.contourf(x, y, stable, levels=[0.5, 1.5], colors=[color], alpha=0.25)
        ax.contour(x, y, stable, levels=[0.5], colors=[color], linewidths=1.5)
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=label))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re $\lambda h$")
    ax.set_ylabel(r"Im $\lambda h$")
    ax.set_title("Absolute stability regions")
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    return ax


def _cobweb(orbit: np.ndarray):
    """Vertices of the cobweb path ``(x_k, x_k) -> (x_k, x_{k+1}) -> (x_{k+1}, x_{k+1})`` through an orbit."""
    doubled = np.repeat(orbit, 2)
    return doubled[:-1], doubled[1:]


def animate_logistic_cobweb(r_values, x0: float = 0.2, n_transient: int = 30, n_keep: int = 40, n_bifurcation: int = 500, interval: int = 120):
    r"""Animate the logistic map's cobweb diagram as the parameter :math:`r` grows.

    Each frame takes the next value of `r`. The left panel draws the
    parabola :math:`f(x) = rx(1-x)`, the diagonal, and the cobweb of the
    orbit from `x0`: its first `n_transient` steps faintly, the following
    `n_keep` steps in red. The red web closes onto a fixed point, then a
    square (period 2), nested squares (period 4, 8, ...), and finally fills
    a band (chaos). The right panel is the bifurcation diagram over the
    range of `r_values`, with the current `r` and its attractor points
    marked. See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.

    Parameters
    ----------
    r_values : array-like of float
        Parameter value of each frame, in ``(0, 4]``.
    x0 : float
        Initial condition in ``(0, 1)``.
    n_transient, n_keep : int
        Orbit steps drawn faintly, then highlighted.
    n_bifurcation : int
        Number of `r` samples in the background bifurcation diagram.
    interval : int
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> anim = animate_logistic_cobweb([2.8, 3.2, 3.5, 3.9], n_bifurcation=50)
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, 300)
    fig, (ax_c, ax_b) = plt.subplots(1, 2, figsize=(11, 5))

    ax_c.plot(xs, xs, color="0.6", lw=1)
    (parabola,) = ax_c.plot([], [], color="k", lw=1.5)
    (transient,) = ax_c.plot([], [], color="tab:blue", lw=0.8, alpha=0.4)
    (attractor,) = ax_c.plot([], [], color="tab:red", lw=1.2)
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("$x_k$")
    ax_c.set_ylabel("$x_{k+1}$")
    title = ax_c.set_title("")

    r_bif = np.linspace(r_values.min(), r_values.max(), n_bifurcation)
    r_plot, x_plot = bifurcation_diagram(r_bif, x0=x0, n_transient=300, n_keep=80)
    plot_bifurcation_diagram(r_plot, x_plot, ax=ax_b)
    marker = ax_b.axvline(r_values[0], color="tab:red", lw=1)
    (points,) = ax_b.plot([], [], "o", color="tab:red", ms=3)
    ax_b.set_ylim(0.0, 1.0)
    fig.tight_layout()

    def update(frame):
        r = r_values[frame]
        orbit = np.concatenate([[x0], LogisticMap(r).iterate(x0, n_transient=0, n_keep=n_transient + n_keep)])
        parabola.set_data(xs, r * xs * (1.0 - xs))
        transient.set_data(*_cobweb(orbit[: n_transient + 1]))
        attractor.set_data(*_cobweb(orbit[n_transient:]))
        marker.set_xdata([r, r])
        points.set_data(np.full(n_keep, r), orbit[-n_keep:])
        title.set_text(f"Cobweb, r = {r:.4f}")
        return [parabola, transient, attractor, marker, points, title]

    return FuncAnimation(fig, update, frames=len(r_values), interval=interval)

"""Plotting helpers for mathematicskit.ode_dynamics: phase portraits, bifurcation
diagrams, Poincare sections, and integrator stability regions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from mathematicskit.integrators.stability import is_absolutely_stable

__all__ = ["plot_phase_portrait", "plot_vector_field", "plot_bifurcation_diagram", "plot_poincare_points", "plot_stability_regions"]


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

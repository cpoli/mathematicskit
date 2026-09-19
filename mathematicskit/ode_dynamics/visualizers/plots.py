"""Plotting helpers for mathematicskit.ode_dynamics: phase portraits, bifurcation
diagrams, and Poincare sections."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_phase_portrait", "plot_vector_field", "plot_bifurcation_diagram", "plot_poincare_points"]


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

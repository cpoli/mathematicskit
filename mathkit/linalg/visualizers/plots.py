"""Plotting helpers for mathkit.linalg: matrix heatmaps and iterative
solver convergence curves.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_matrix_heatmap", "plot_residual_history", "plot_eigenvalue_spectrum"]


def plot_matrix_heatmap(a: np.ndarray, ax=None, title: str = "", cmap: str = "coolwarm"):
    """Heatmap of a matrix's entries, e.g. to visualize an LU/QR factor.

    Parameters
    ----------
    a : ndarray, shape (m, n)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    title : str
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    a = np.asarray(a, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    vmax = np.max(np.abs(a)) or 1.0
    ax.imshow(a, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    return ax


def plot_residual_history(result, ax=None, label=None):
    """Semilog plot of an iterative solver's residual norm vs. iteration.

    Parameters
    ----------
    result : IterativeSolveResult
        From :class:`~mathkit.linalg.systems.iterative.ConjugateGradient`
        or :class:`~mathkit.linalg.systems.iterative.GMRES`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional
        Legend label; defaults to ``result.method``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    residuals = np.where(result.residual_history == 0.0, np.nan, result.residual_history)
    ax.semilogy(np.arange(len(residuals)), residuals, marker="o", ms=3, label=label or result.method)
    ax.set_xlabel("iteration")
    ax.set_ylabel("||r_k||")
    ax.set_title("Iterative solver convergence")
    return ax


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, ax=None):
    """Scatter plot of eigenvalues on the real line (or complex plane).

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    eigenvalues = np.asarray(eigenvalues)
    if ax is None:
        _, ax = plt.subplots()
    if np.iscomplexobj(eigenvalues):
        ax.scatter(eigenvalues.real, eigenvalues.imag)
        ax.set_xlabel("Re(lambda)")
        ax.set_ylabel("Im(lambda)")
    else:
        ax.scatter(eigenvalues, np.zeros_like(eigenvalues))
        ax.set_xlabel("lambda")
        ax.set_yticks([])
    ax.set_title("Eigenvalue spectrum")
    return ax

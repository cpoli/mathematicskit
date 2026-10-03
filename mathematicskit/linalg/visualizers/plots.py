"""Plotting helpers for mathematicskit.linalg: matrix heatmaps, iterative
solver convergence curves, and an animation of power iteration.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

__all__ = ["plot_matrix_heatmap", "plot_residual_history", "plot_eigenvalue_spectrum", "animate_power_iteration"]


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
        From :class:`~mathematicskit.linalg.systems.iterative.ConjugateGradient`
        or :class:`~mathematicskit.linalg.systems.iterative.GMRES`.
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


def animate_power_iteration(a: np.ndarray, result, interval: int = 500):
    r"""Animate power iteration turning a vector towards the dominant eigenvector.

    For a 2x2 matrix the left panel draws the iterate :math:`v_k` as an
    arrow in the plane, with the eigenvector directions as lines; for a
    larger matrix it draws the components of :math:`v_k` as bars against
    the dominant eigenvector's components. The right panel grows the angle
    :math:`\theta_k` between :math:`v_k` and the dominant eigenvector,
    beside the predicted :math:`\tan\theta_k = \tan\theta_0\,|\lambda_2/\lambda_1|^k`
    (exact for a symmetric matrix, asymptotic in general). Since
    :math:`v_k` and :math:`-v_k` span the same line, the sign flips caused
    by a negative dominant eigenvalue are factored out.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        The matrix that was iterated.
    result : EigenResult
        From :func:`~mathematicskit.linalg.systems.eigen.power_iteration`
        (needs ``result.extra["vectors"]``).
    interval : int
        Delay between frames, in milliseconds.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        One frame per iterate :math:`v_0, \dots, v_k`.

    Examples
    --------
    >>> import numpy as np
    >>> from mathematicskit.linalg import power_iteration
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> anim = animate_power_iteration(A, power_iteration(A, tol=1e-6))
    >>> html = anim.to_jshtml()  # self-contained HTML/JS player, no ffmpeg needed
    """
    if "vectors" not in result.extra:
        raise ValueError("result has no iterate history; use power_iteration()")
    a = np.asarray(a, dtype=np.float64)
    vectors = np.asarray(result.extra["vectors"])
    n = a.shape[0]
    lams, vecs = np.linalg.eig(a)
    order = np.argsort(-np.abs(lams))
    lams, vecs = lams[order], vecs[:, order]
    q = np.real(vecs[:, 0])
    q = q / np.linalg.norm(q)
    signs = np.where(vectors @ q < 0, -1.0, 1.0)
    aligned = vectors * signs[:, None]
    cosines = np.clip(aligned @ q, -1.0, 1.0)
    angles = np.arccos(cosines)
    rate = abs(lams[1] / lams[0]) if n > 1 else 0.0
    ks = np.arange(len(vectors))
    predicted = np.arctan(np.tan(angles[0]) * rate**ks)

    fig, (ax_v, ax_a) = plt.subplots(1, 2, figsize=(11, 4.8))
    if n == 2:
        circle = np.linspace(0.0, 2.0 * np.pi, 200)
        ax_v.plot(np.cos(circle), np.sin(circle), color="0.8", lw=1)
        for j in range(2):
            e = np.real(vecs[:, j])
            e = 1.3 * e / np.linalg.norm(e)
            color = "tab:green" if j == 0 else "0.5"
            ax_v.plot([-e[0], e[0]], [-e[1], e[1]], "--", color=color, lw=1.2, label=rf"eigenvector, $\lambda = {np.real(lams[j]):.3g}$")
        ax_v.set_xlim(-1.35, 1.35)
        ax_v.set_ylim(-1.35, 1.35)
        ax_v.set_aspect("equal")
        ax_v.set_xlabel("$x_1$")
        ax_v.set_ylabel("$x_2$")
        (trail,) = ax_v.plot([], [], "o", color="tab:blue", alpha=0.3, ms=4)
        (arrow,) = ax_v.plot([], [], "-", color="tab:red", lw=2.5, solid_capstyle="round")
        (tip,) = ax_v.plot([], [], "o", color="tab:red", ms=7, label="$v_k$")
        ax_v.legend(fontsize=8, loc="lower right")

        def draw_vector(k):
            v = vectors[k]
            trail.set_data(vectors[:k, 0], vectors[:k, 1])
            arrow.set_data([0.0, v[0]], [0.0, v[1]])
            tip.set_data([v[0]], [v[1]])
            return [trail, arrow, tip]

    else:
        idx = np.arange(n)
        bars = ax_v.bar(idx, aligned[0], color="tab:red", alpha=0.7, label=r"$\pm v_k$")
        ax_v.plot(idx, q, "_", color="tab:green", ms=20, mew=3, label="dominant eigenvector")
        ax_v.axhline(0.0, color="0.6", lw=0.8)
        ax_v.set_ylim(-1.05, 1.05)
        ax_v.set_xticks(idx)
        ax_v.set_xlabel("component")
        ax_v.legend(fontsize=8, loc="lower right")

        def draw_vector(k):
            for bar, height in zip(bars, aligned[k], strict=True):
                bar.set_height(height)
            return list(bars)

    ax_a.set_yscale("log")
    floor = max(angles[angles > 0].min() if np.any(angles > 0) else 1e-16, 1e-16)
    ax_a.set_xlim(-0.5, len(vectors) - 0.5)
    ax_a.set_ylim(floor / 3.0, np.pi)
    if n > 1:
        ax_a.plot(ks, np.maximum(predicted, floor / 3.0), "k--", lw=1, label=rf"$\tan\theta_0\,|\lambda_2/\lambda_1|^k$, ratio {rate:.3g}")
    (angle_line,) = ax_a.plot([], [], "o-", color="tab:red", ms=4, label=r"angle $\theta_k$")
    ax_a.set_xlabel("iteration k")
    ax_a.set_ylabel("angle to dominant eigenvector (rad)")
    ax_a.legend(fontsize=8)
    title = fig.suptitle("")
    fig.tight_layout()

    def update(k):
        angle_line.set_data(ks[: k + 1], np.where(angles[: k + 1] > 0, angles[: k + 1], np.nan))
        title.set_text(rf"Power iteration, k = {k}: Rayleigh quotient {vectors[k] @ a @ vectors[k]:.6g}")
        return [*draw_vector(k), angle_line, title]

    return FuncAnimation(fig, update, frames=len(vectors), interval=interval)

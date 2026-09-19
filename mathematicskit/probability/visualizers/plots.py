"""Plotting helpers for mathematicskit.probability: PMF/PDF bars/curves, CLT
histograms, and Markov-chain transition-matrix heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

__all__ = ["plot_distribution", "plot_clt_histogram", "plot_transition_matrix"]


def plot_distribution(dist, ax=None, x_range=None, n_points: int = 200, label=None):
    """Plot a distribution's PMF (stem plot) or PDF (curve).

    Parameters
    ----------
    dist : DiscreteDistribution or ContinuousDistribution
        Any :mod:`mathematicskit.probability` distribution.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    x_range : tuple of float, optional
        ``(min, max)``; defaults to +-4 standard deviations around the mean.
    n_points : int
        Number of points for a continuous distribution's curve.
    label : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if x_range is None:
        x_range = (dist.mean - 4.0 * dist.std, dist.mean + 4.0 * dist.std)
    if hasattr(dist, "pmf"):
        ks = np.arange(int(np.floor(x_range[0])), int(np.ceil(x_range[1])) + 1)
        ks = ks[ks >= 0]
        ax.stem(ks, dist.pmf(ks), basefmt=" ", label=label)
        ax.set_ylabel("P(X = k)")
    else:
        xs = np.linspace(x_range[0], x_range[1], n_points)
        ax.plot(xs, dist.pdf(xs), label=label)
        ax.set_ylabel("f(x)")
    ax.set_xlabel("x")
    ax.set_title(type(dist).__name__)
    if label:
        ax.legend()
    return ax


def plot_clt_histogram(standardized_means: np.ndarray, ax=None, n_bins: int = 40):
    """Histogram of standardized sample means, with the standard normal density overlaid.

    Parameters
    ----------
    standardized_means : ndarray
        From :func:`~mathematicskit.probability.systems.limit_theorems.central_limit_theorem_sample_means`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    n_bins : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(standardized_means, bins=n_bins, density=True, alpha=0.6, color="steelblue", label="sample means")
    xs = np.linspace(-4.0, 4.0, 200)
    ax.plot(xs, stats.norm.pdf(xs), color="firebrick", lw=2, label="N(0, 1)")
    ax.set_xlabel("standardized sample mean")
    ax.set_ylabel("density")
    ax.set_title("Central Limit Theorem")
    ax.legend()
    return ax


def plot_transition_matrix(chain, ax=None, cmap: str = "viridis"):
    """Heatmap of a Markov chain's transition matrix.

    Parameters
    ----------
    chain : MarkovChain
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    cmap : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(chain.p, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_title("Transition matrix")
    return ax

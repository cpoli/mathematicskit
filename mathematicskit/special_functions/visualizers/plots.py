"""Plotting helpers for mathematicskit.special_functions: orthogonal polynomial
families, and the FFT speed comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_polynomial_family", "plot_fft_timing_comparison"]


def plot_polynomial_family(poly_func, degrees, x_range=(-1.0, 1.0), ax=None):
    """Plot several degrees of an orthogonal polynomial family on the same axes.

    Parameters
    ----------
    poly_func : callable
        ``poly_func(n, x) -> ndarray`` (e.g.
        :func:`~mathematicskit.special_functions.systems.orthogonal_polynomials.legendre_polynomial`).
    degrees : sequence of int
    x_range : tuple of float
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    xs = np.linspace(x_range[0], x_range[1], 300)
    for n in degrees:
        ax.plot(xs, poly_func(n, xs), label=f"n={n}")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title(poly_func.__name__)
    ax.legend()
    return ax


def plot_fft_timing_comparison(sizes, ax=None, seed: int = 0):
    """Log-log plot of naive-DFT/radix-2-FFT/numpy.fft timing vs. signal length.

    Parameters
    ----------
    sizes : sequence of int
        Signal lengths to time (each must be a power of 2).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    seed : int

    Returns
    -------
    matplotlib.axes.Axes
    """
    from mathematicskit.special_functions.systems.fourier_transform import compare_fft_methods

    if ax is None:
        _, ax = plt.subplots()
    naive_times, radix2_times, numpy_times = [], [], []
    for n in sizes:
        result = compare_fft_methods(n, seed=seed)
        naive_times.append(result.naive_time)
        radix2_times.append(result.radix2_time)
        numpy_times.append(result.numpy_time)
    ax.loglog(sizes, naive_times, "o-", label="naive DFT: O(n^2)")
    ax.loglog(sizes, radix2_times, "o-", label="radix-2 FFT: O(n log n)")
    ax.loglog(sizes, numpy_times, "o-", label="numpy.fft")
    ax.set_xlabel("n")
    ax.set_ylabel("time (s)")
    ax.set_title("FFT method timing comparison")
    ax.legend()
    return ax

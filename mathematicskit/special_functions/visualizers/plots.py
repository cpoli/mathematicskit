"""Plotting helpers for mathematicskit.special_functions: orthogonal polynomial
families, the FFT speed comparison, filter frequency responses, pole-zero
diagrams, wavelet decompositions, and scalograms."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_polynomial_family",
    "plot_fft_timing_comparison",
    "plot_frequency_response",
    "plot_pole_zero",
    "plot_wavelet_decomposition",
    "plot_scalogram",
]


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


def plot_frequency_response(response, ax=None, db: bool = True, label: Optional[str] = None):
    """Plot a filter's magnitude response.

    Parameters
    ----------
    response : FrequencyResponseResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    db : bool
        Plot :math:`20\\log_{10}|H|` rather than :math:`|H|`.
    label : str, optional
        Legend label for this curve.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if db:
        ax.plot(response.frequencies, 20.0 * np.log10(np.maximum(response.magnitude, 1e-12)), label=label)
        ax.set_ylabel("|H| (dB)")
    else:
        ax.plot(response.frequencies, response.magnitude, label=label)
        ax.set_ylabel("|H|")
    ax.set_xlabel("frequency")
    ax.set_title("Frequency response")
    if label is not None:
        ax.legend()
    return ax


def plot_pole_zero(result, ax=None):
    """Pole-zero diagram in the z-plane, with the unit circle.

    Parameters
    ----------
    result : PoleZeroResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, label="unit circle")
    ax.plot(result.zeros.real, result.zeros.imag, "o", mfc="none", ms=9, label="zeros")
    ax.plot(result.poles.real, result.poles.imag, "x", ms=9, mew=2, label="poles")
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.axvline(0.0, color="gray", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title("Pole-zero diagram (" + ("stable" if result.is_stable else "unstable") + ")")
    ax.legend()
    return ax


def plot_wavelet_decomposition(result, ax=None):
    """Stem-style plot of each level's wavelet coefficients, stacked from finest to coarsest.

    Parameters
    ----------
    result : WaveletDecompositionResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    rows = [(f"d{i + 1}", d) for i, d in enumerate(result.details)] + [(f"a{len(result.details)}", result.approximation)]
    n = 2 * len(result.details[0])
    for offset, (name, coeffs) in enumerate(rows):
        scale = np.max(np.abs(coeffs)) or 1.0
        positions = (np.arange(len(coeffs)) + 0.5) * n / len(coeffs)
        ax.vlines(positions, -offset, -offset + 0.45 * np.asarray(coeffs) / scale)
        ax.text(-0.02 * n, -offset, name, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlim(-0.08 * n, n)
    ax.set_xlabel("sample")
    ax.set_title(f"Wavelet decomposition ({result.wavelet})")
    return ax


def plot_scalogram(result, ax=None):
    """Plot the power :math:`|W(s, t)|^2` of a continuous wavelet transform against time and frequency.

    Parameters
    ----------
    result : ScalogramResult
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.pcolormesh(result.times, result.frequencies, np.abs(result.coefficients) ** 2, shading="auto")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")
    ax.set_title("Morlet scalogram")
    return ax

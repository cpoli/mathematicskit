"""Smoke tests for mathematicskit.special_functions.visualizers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np

from mathematicskit.special_functions.systems.filters import butterworth_filter, frequency_response
from mathematicskit.special_functions.systems.orthogonal_polynomials import legendre_polynomial
from mathematicskit.special_functions.systems.wavelets import discrete_wavelet_transform, morlet_cwt
from mathematicskit.special_functions.systems.z_transform import poles_zeros
from mathematicskit.special_functions.visualizers.plots import (
    plot_fft_timing_comparison,
    plot_frequency_response,
    plot_pole_zero,
    plot_polynomial_family,
    plot_scalogram,
    plot_wavelet_decomposition,
)


def test_plot_polynomial_family_returns_axes():
    ax = plot_polynomial_family(legendre_polynomial, degrees=[0, 1, 2, 3])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_fft_timing_comparison_returns_axes():
    ax = plot_fft_timing_comparison([32, 64, 128])
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_frequency_response_returns_axes():
    ax = plot_frequency_response(frequency_response(butterworth_filter(4, 0.3)), label="N=4")
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_pole_zero_returns_axes():
    filt = butterworth_filter(4, 0.3)
    assert isinstance(plot_pole_zero(poles_zeros(filt.b, filt.a)), matplotlib.axes.Axes)


def test_plot_wavelet_decomposition_returns_axes():
    result = discrete_wavelet_transform(np.random.default_rng(0).normal(size=64), "db2", level=3)
    assert isinstance(plot_wavelet_decomposition(result), matplotlib.axes.Axes)


def test_plot_scalogram_returns_axes():
    result = morlet_cwt(np.sin(np.arange(256) * 0.3), np.geomspace(2.0, 40.0, 20))
    assert isinstance(plot_scalogram(result), matplotlib.axes.Axes)

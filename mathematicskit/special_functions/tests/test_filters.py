"""Tests for Butterworth, Chebyshev, and window-method FIR filter design
against their closed-form magnitude responses."""

import numpy as np
import pytest
from scipy import signal

from mathematicskit.special_functions.core.base import FilterCoefficients
from mathematicskit.special_functions.systems.filters import (
    apply_filter,
    butterworth_filter,
    chebyshev1_filter,
    fir_window_filter,
    frequency_response,
)


@pytest.mark.parametrize("order", [1, 2, 4, 7])
def test_butterworth_matches_bilinear_transformed_closed_form(order):
    """|H|^2 = 1 / (1 + (tan(w/2) / tan(wc/2))^(2N)) after the bilinear transform."""
    wc = 0.3 * np.pi
    filt = butterworth_filter(order, 0.3)
    w = np.linspace(0.01, 0.99 * np.pi, 50)
    response = frequency_response(filt, worN=w)
    expected = 1.0 / (1.0 + (np.tan(w / 2) / np.tan(wc / 2)) ** (2 * order))
    np.testing.assert_allclose(response.magnitude**2, expected, atol=1e-9)


def test_butterworth_highpass_blocks_dc_and_passes_nyquist():
    response = frequency_response(butterworth_filter(3, 0.4, btype="highpass"), worN=[0.0, np.pi])
    assert response.magnitude[0] == pytest.approx(0.0, abs=1e-12)
    assert response.magnitude[1] == pytest.approx(1.0)


def test_cutoff_in_hz_with_sampling_rate():
    response = frequency_response(butterworth_filter(4, 100.0, fs=1000.0), worN=[100.0], fs=1000.0)
    assert response.magnitude[0] ** 2 == pytest.approx(0.5)


def test_chebyshev_passband_ripple_bounds():
    filt = chebyshev1_filter(6, 0.5, 0.4)
    passband = frequency_response(filt, worN=np.linspace(0.0, 0.4 * np.pi, 400)).magnitude
    assert passband.max() <= 1.0 + 1e-9
    assert passband.min() >= 10 ** (-0.5 / 20) - 1e-9


def test_fir_window_filter_is_linear_phase_with_unit_dc_gain():
    filt = fir_window_filter(51, 0.3, window=("kaiser", 6.0))
    np.testing.assert_allclose(filt.b, filt.b[::-1])
    np.testing.assert_allclose(filt.a, [1.0])
    assert filt.b.sum() == pytest.approx(1.0)
    w = np.linspace(0.01, 0.25 * np.pi, 20)
    response = frequency_response(filt, worN=w)
    np.testing.assert_allclose(np.angle(response.response * np.exp(1j * w * 25)), 0.0, atol=1e-6)


def test_apply_filter_solves_difference_equation():
    filt = FilterCoefficients(b=np.array([1.0]), a=np.array([1.0, -0.5]))
    impulse = np.zeros(6)
    impulse[0] = 1.0
    np.testing.assert_allclose(apply_filter(filt, impulse), 0.5 ** np.arange(6))


def test_lowpass_removes_high_frequency_tone():
    t = np.arange(2000) / 1000.0
    low, high = np.sin(2 * np.pi * 5.0 * t), np.sin(2 * np.pi * 200.0 * t)
    filt = butterworth_filter(6, 50.0, fs=1000.0)
    y = apply_filter(filt, low + high, zero_phase=True)
    np.testing.assert_allclose(y[200:-200], low[200:-200], atol=1e-3)


def test_frequency_response_matches_scipy_freqz():
    filt = butterworth_filter(3, 0.2)
    w, h = signal.freqz(filt.b, filt.a, worN=64)
    response = frequency_response(filt, worN=64)
    np.testing.assert_allclose(response.frequencies, w)
    np.testing.assert_allclose(response.response, h)

"""Tests for direct, FFT, and circular convolution and cross-correlation
against closed-form results and the convolution theorem."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.convolution import (
    circular_convolve,
    compare_convolution_methods,
    convolve_direct,
    convolve_fft,
    cross_correlate,
)


def test_convolution_of_polynomial_coefficients_is_polynomial_product():
    p, q = np.array([1.0, 2.0, 3.0]), np.array([4.0, -1.0])
    np.testing.assert_allclose(convolve_direct(p, q), np.polymul(p, q))


def test_fft_convolution_matches_direct():
    rng = np.random.default_rng(0)
    x, h = rng.normal(size=300), rng.normal(size=41)
    for mode in ("full", "same", "valid"):
        np.testing.assert_allclose(convolve_fft(x, h, mode), convolve_direct(x, h, mode), atol=1e-10)


def test_convolution_with_unit_impulse_is_identity():
    x = np.arange(6.0)
    np.testing.assert_allclose(convolve_direct(x, [1.0]), x)


def test_circular_convolution_equals_explicit_modular_sum():
    rng = np.random.default_rng(1)
    x, h = rng.normal(size=16), rng.normal(size=16)
    expected = np.array([sum(x[k] * h[(n - k) % 16] for k in range(16)) for n in range(16)])
    np.testing.assert_allclose(circular_convolve(x, h), expected, atol=1e-12)


def test_circular_convolution_equals_wrapped_linear_convolution():
    rng = np.random.default_rng(2)
    x, h = rng.normal(size=8), rng.normal(size=8)
    linear = np.convolve(x, h)
    wrapped = linear[:8] + np.concatenate([linear[8:], [0.0]])
    np.testing.assert_allclose(circular_convolve(x, h), wrapped, atol=1e-12)


def test_circular_convolution_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        circular_convolve(np.zeros(4), np.zeros(5))


def test_cross_correlation_peak_recovers_shift():
    rng = np.random.default_rng(3)
    template = rng.normal(size=50)
    x = np.concatenate([np.zeros(37), template, np.zeros(20)])
    c = cross_correlate(x, template)
    assert int(np.argmax(c)) - (len(template) - 1) == 37


def test_compare_convolution_methods_agree():
    result = compare_convolution_methods(2000, 500, seed=0)
    assert result.max_error < 1e-9
    assert result.direct_result.shape == (2499,)

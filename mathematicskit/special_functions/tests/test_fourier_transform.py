"""Tests for the naive DFT, radix-2 FFT, and numpy.fft comparison
against closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.fourier_transform import compare_fft_methods, dft_naive, fft_numpy, fft_radix2


def test_dft_naive_matches_numpy_fft():
    rng = np.random.default_rng(0)
    x = rng.normal(size=17)  # deliberately not a power of 2
    np.testing.assert_allclose(dft_naive(x), np.fft.fft(x), atol=1e-8)


def test_fft_radix2_matches_numpy_fft():
    rng = np.random.default_rng(1)
    x = rng.normal(size=64)
    np.testing.assert_allclose(fft_radix2(x), np.fft.fft(x), atol=1e-8)


def test_fft_radix2_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft_radix2(np.zeros(10))


def test_fft_numpy_matches_reference():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(fft_numpy(x), np.fft.fft(x))


def test_dft_of_constant_signal_is_delta_at_zero_frequency():
    x = np.ones(8)
    result = dft_naive(x)
    assert result[0] == pytest.approx(8.0)
    np.testing.assert_allclose(result[1:], 0.0, atol=1e-8)


def test_compare_fft_methods_cross_checks_agree():
    result = compare_fft_methods(128, seed=0)
    assert result.max_error_naive_vs_numpy < 1e-8
    assert result.max_error_radix2_vs_numpy < 1e-8


def test_compare_fft_methods_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        compare_fft_methods(100)


def test_radix2_faster_than_naive_for_large_n():
    """Not a strict timing guarantee (machine-dependent), but the
    O(n^2) vs O(n log n) gap should be visible at a large enough n."""
    result = compare_fft_methods(4096, seed=0)
    assert result.radix2_time < result.naive_time

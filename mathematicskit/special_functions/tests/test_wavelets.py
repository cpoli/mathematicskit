"""Tests for Daubechies filters, the discrete wavelet transform, and the
Morlet continuous wavelet transform against their defining properties."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.wavelets import (
    daubechies_filter,
    discrete_wavelet_transform,
    inverse_discrete_wavelet_transform,
    morlet_cwt,
    wavelet_filters,
)


def test_haar_filter_closed_form():
    np.testing.assert_allclose(daubechies_filter(1), [2**-0.5, 2**-0.5])


def test_db2_matches_daubechies_closed_form():
    r3 = np.sqrt(3.0)
    expected = np.array([1 + r3, 3 + r3, 3 - r3, 1 - r3]) / (4 * np.sqrt(2.0))
    np.testing.assert_allclose(daubechies_filter(2), expected, atol=1e-14)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 8])
def test_daubechies_orthonormality_and_vanishing_moments(p):
    """sum h = sqrt 2, sum h[k] h[k + 2m] = delta_m, and the highpass kills polynomials of degree < p."""
    h, g = wavelet_filters(f"db{p}")
    assert h.size == 2 * p
    assert h.sum() == pytest.approx(np.sqrt(2.0))
    for m in range(p):
        assert np.dot(h[2 * m :], h[: h.size - 2 * m]) == pytest.approx(1.0 if m == 0 else 0.0, abs=1e-9)
    k = np.arange(h.size)
    for moment in range(p):
        assert np.sum(g * k**moment) == pytest.approx(0.0, abs=1e-6 * h.size**moment)


def test_unknown_wavelet_rejected():
    with pytest.raises(ValueError):
        wavelet_filters("sym4")


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
def test_dwt_is_energy_preserving_and_perfectly_reconstructs(wavelet):
    x = np.random.default_rng(0).normal(size=256)
    result = discrete_wavelet_transform(x, wavelet, level=4)
    energy = np.sum(result.approximation**2) + sum(np.sum(d**2) for d in result.details)
    assert energy == pytest.approx(np.sum(x**2))
    assert [d.size for d in result.details] == [128, 64, 32, 16]
    np.testing.assert_allclose(inverse_discrete_wavelet_transform(result), x, atol=1e-10)


def test_haar_one_level_is_pairwise_sums_and_differences():
    x = np.array([4.0, 2.0, 5.0, 5.0])
    result = discrete_wavelet_transform(x, "haar", level=1)
    np.testing.assert_allclose(result.approximation, np.array([6.0, 10.0]) / np.sqrt(2))
    np.testing.assert_allclose(result.details[0], np.array([2.0, 0.0]) / np.sqrt(2), atol=1e-12)


def test_db2_details_vanish_on_linear_signal_interior():
    """Two vanishing moments: detail coefficients of a line are zero away from the periodic wrap."""
    result = discrete_wavelet_transform(np.arange(64.0), "db2", level=1)
    np.testing.assert_allclose(result.details[0][:-1], 0.0, atol=1e-10)


def test_default_level_and_length_validation():
    assert len(discrete_wavelet_transform(np.zeros(64), "haar").details) == 6
    with pytest.raises(ValueError):
        discrete_wavelet_transform(np.zeros(12), "haar", level=3)


def test_morlet_scalogram_peaks_at_signal_frequency():
    dt = 0.005
    t = np.arange(2048) * dt
    scales = np.geomspace(0.02, 1.0, 120)
    for f0 in (4.0, 20.0):
        result = morlet_cwt(np.cos(2 * np.pi * f0 * t), scales, dt=dt)
        power = np.mean(np.abs(result.coefficients[:, 400:-400]) ** 2, axis=1)
        assert result.frequencies[np.argmax(power)] == pytest.approx(f0, rel=0.05)


def test_morlet_cwt_localizes_frequency_change_in_time():
    dt = 0.01
    t = np.arange(2000) * dt
    x = np.where(t < 10.0, np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 8.0 * t))
    scales = np.geomspace(0.05, 1.0, 80)
    result = morlet_cwt(x, scales, dt=dt)
    power = np.abs(result.coefficients) ** 2
    early, late = power[:, 300:700].mean(axis=1), power[:, 1300:1700].mean(axis=1)
    assert result.frequencies[np.argmax(early)] == pytest.approx(2.0, rel=0.1)
    assert result.frequencies[np.argmax(late)] == pytest.approx(8.0, rel=0.1)
    assert result.coefficients.shape == (80, 2000)

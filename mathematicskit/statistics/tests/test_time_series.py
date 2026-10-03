"""Tests for sample autocorrelation and Yule-Walker AR fitting."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.time_series import ar_autocorrelation, autocorrelation, simulate_ar, yule_walker


def test_autocorrelation_lag_zero_and_white_noise():
    x = np.random.default_rng(0).standard_normal(20000)
    acf = autocorrelation(x, 10)
    assert acf[0] == 1.0
    assert np.all(np.abs(acf[1:]) < 3 / np.sqrt(x.size))  # Bartlett's 1/sqrt(n) band


def test_ar1_autocorrelation_is_geometric():
    assert np.allclose(ar_autocorrelation([0.7], 5), 0.7 ** np.arange(6))
    x = simulate_ar([0.7], n=100000, seed=2)
    assert np.allclose(autocorrelation(x, 5), 0.7 ** np.arange(6), atol=0.02)


def test_ar2_theoretical_acf_satisfies_yule_walker():
    phi = np.array([0.75, -0.5])
    rho = ar_autocorrelation(phi, 8)
    assert rho[1] == pytest.approx(phi[0] / (1 - phi[1]))  # closed form for AR(2)
    for k in range(2, 9):
        assert rho[k] == pytest.approx(phi[0] * rho[k - 1] + phi[1] * rho[k - 2])


def test_yule_walker_recovers_coefficients_and_noise_variance():
    phi = [0.6, -0.2, 0.1]
    x = simulate_ar(phi, n=100000, sigma=2.0, mean=5.0, seed=3)
    result = yule_walker(x, 3)
    assert np.allclose(result.coefficients, phi, atol=0.02)
    assert result.noise_variance == pytest.approx(4.0, rel=0.03)
    assert result.mean == pytest.approx(5.0, abs=0.05)
    assert result.autocorrelation.shape == (4,)


def test_overfitted_order_gives_near_zero_extra_coefficient():
    x = simulate_ar([0.5], n=50000, seed=4)
    assert abs(yule_walker(x, 2).coefficients[1]) < 0.02


def test_simulated_ar1_variance():
    x = simulate_ar([-0.8], n=200000, sigma=1.5, seed=5)
    assert x.var() == pytest.approx(1.5**2 / (1 - 0.64), rel=0.03)

"""Tests for Euler-Maruyama, geometric Brownian motion, and the Ornstein-Uhlenbeck process."""

import numpy as np
import pytest

from mathematicskit.probability.systems.sde import euler_maruyama, geometric_brownian_motion, ornstein_uhlenbeck


def test_zero_diffusion_reduces_to_euler_method():
    result = euler_maruyama(lambda x, t: -x, lambda x, t: np.zeros_like(x), 1.0, t_max=1.0, n_steps=10)
    assert result.paths[0, -1] == pytest.approx(0.9**10)


def test_pure_noise_reproduces_wiener_path():
    result = euler_maruyama(lambda x, t: np.zeros_like(x), lambda x, t: np.ones_like(x), 0.0, n_steps=100, n_paths=3, seed=5)
    assert np.allclose(result.paths, result.wiener)
    assert result.paths.shape == result.wiener.shape == (3, 101)


def test_gbm_exact_moments():
    mu, sigma, t = 0.05, 0.3, 2.0
    result = geometric_brownian_motion(mu, sigma, x0=2.0, t_max=t, n_steps=20, n_paths=100000, seed=1)
    x_t = result.paths[:, -1]
    assert x_t.mean() == pytest.approx(2.0 * np.exp(mu * t), rel=0.01)
    assert np.log(x_t / 2.0).mean() == pytest.approx((mu - 0.5 * sigma**2) * t, abs=0.01)  # the Itô correction
    assert np.log(x_t).var() == pytest.approx(sigma**2 * t, rel=0.02)


def test_euler_maruyama_strong_order_one_half():
    mu, sigma, t_max, fine = 1.5, 1.0, 1.0, 2**10
    exact = geometric_brownian_motion(mu, sigma, t_max=t_max, n_steps=fine, n_paths=2000, seed=7)
    dw_fine = np.diff(exact.wiener, axis=1)
    errors, dts = [], []
    for r in (16, 32, 64, 128):
        dw = dw_fine.reshape(dw_fine.shape[0], -1, r).sum(axis=2)
        em = euler_maruyama(lambda x, t: mu * x, lambda x, t: sigma * x, 1.0, t_max, fine // r, dw=dw)
        errors.append(np.mean(np.abs(em.paths[:, -1] - exact.paths[:, -1])))
        dts.append(t_max * r / fine)
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert slope == pytest.approx(0.5, abs=0.15)


def test_gbm_euler_maruyama_shares_brownian_path_with_exact():
    em = geometric_brownian_motion(0.1, 0.2, n_steps=4000, n_paths=5, seed=3, method="euler_maruyama")
    exact = geometric_brownian_motion(0.1, 0.2, n_steps=4000, n_paths=5, seed=3)
    assert em.method == "euler_maruyama" and exact.method == "exact"
    assert np.allclose(em.wiener, exact.wiener)
    assert np.max(np.abs(em.paths[:, -1] - exact.paths[:, -1])) < 0.01


def test_gbm_unknown_method():
    with pytest.raises(ValueError):
        geometric_brownian_motion(0.1, 0.2, method="milstein")


def test_ornstein_uhlenbeck_mean_and_variance():
    theta, mu, sigma, x0, t = 1.0, -1.0, 0.8, 2.0, 0.7
    result = ornstein_uhlenbeck(theta, mu, sigma, x0=x0, t_max=t, n_steps=700, n_paths=40000, seed=2)
    x_t = result.paths[:, -1]
    assert x_t.mean() == pytest.approx(mu + (x0 - mu) * np.exp(-theta * t), abs=0.01)
    assert x_t.var() == pytest.approx(sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t)), rel=0.03)

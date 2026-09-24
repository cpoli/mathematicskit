"""Tests for the Robbins-Monro iteration."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.stochastic import robbins_monro


def test_unit_gain_on_quadratic_is_the_running_sample_mean():
    # With a_k = 1/(k+1) and Y = x - mu + e_k, x_n is exactly the mean of mu - e_k.
    mu = np.array([1.0, -2.0])
    n = 500
    result = robbins_monro(lambda x, rng: x - mu + rng.normal(size=2), [10.0, 10.0], n_iter=n, seed=7)
    noise = np.random.default_rng(7).normal(size=(n, 2))
    np.testing.assert_allclose(result.x, mu - noise.mean(axis=0), atol=1e-12)
    assert result.path.shape == (n + 1, 2)


def test_converges_to_root_of_noisy_regression_function():
    # Root of M(x) = 2(x - 5) observed with noise; error shrinks like 1/sqrt(n).
    result = robbins_monro(lambda x, rng: 2.0 * (x - 5.0) + rng.normal(size=x.shape), [0.0], a0=0.5, n_iter=50000, seed=1, f=lambda x: float((x[0] - 5.0) ** 2))
    assert result.x[0] == pytest.approx(5.0, abs=0.02)
    assert result.fun == pytest.approx((result.x[0] - 5.0) ** 2)


def test_noise_free_problem_is_deterministic_and_fun_nan_without_f():
    result = robbins_monro(lambda x, rng: x, [4.0], n_iter=3)
    # x1 = 4 - 4 = 0 with a_0 = 1, then stays at 0.
    np.testing.assert_allclose(result.path[:, 0], [4.0, 0.0, 0.0, 0.0])
    assert np.isnan(result.fun)

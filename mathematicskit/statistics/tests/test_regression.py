"""Tests for OLS linear regression against closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.regression import linear_regression


def test_recovers_exact_coefficients_for_noiseless_data():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0
    result = linear_regression(x, y)
    np.testing.assert_allclose(result.coefficients, [1.0, 2.0], atol=1e-8)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)


def test_residuals_are_small_for_lightly_noisy_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 200)
    y = 3.0 * x - 2.0 + rng.normal(0.0, 0.1, x.shape)
    result = linear_regression(x, y)
    assert np.max(np.abs(result.residuals)) < 1.0
    assert result.r_squared > 0.99


def test_no_intercept_fit_forces_line_through_origin():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    result = linear_regression(x, y, add_intercept=False)
    np.testing.assert_allclose(result.coefficients, [2.0], atol=1e-8)


def test_standard_errors_shrink_with_more_data():
    rng = np.random.default_rng(1)
    x_small = rng.uniform(-1, 1, 10)
    y_small = 2.0 * x_small + rng.normal(0, 1.0, 10)
    x_large = rng.uniform(-1, 1, 1000)
    y_large = 2.0 * x_large + rng.normal(0, 1.0, 1000)
    small_result = linear_regression(x_small, y_small)
    large_result = linear_regression(x_large, y_large)
    assert large_result.standard_errors[1] < small_result.standard_errors[1]


def test_t_statistics_and_p_values_have_consistent_shapes():
    x = np.linspace(0, 1, 20)
    y = x + np.sin(x)
    result = linear_regression(x, y)
    assert result.t_statistics.shape == (2,)
    assert result.p_values.shape == (2,)
    assert np.all((result.p_values >= 0.0) & (result.p_values <= 1.0))


def test_perfect_fit_gives_significant_slope_p_value():
    x = np.linspace(0, 1, 50)
    y = 5.0 * x + 2.0 + np.random.default_rng(2).normal(0, 0.01, 50)
    result = linear_regression(x, y)
    assert result.p_values[1] < 0.001

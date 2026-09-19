"""Tests for least-squares polynomial regression against closed-form
properties: exact recovery of noiseless polynomial data, R^2 = 1 for a
perfect fit, and worsening condition number at higher degree."""

import numpy as np
import pytest

from mathkit.numerical_analysis.systems.regression import PolynomialRegression


def test_recovers_exact_coefficients_for_noiseless_data():
    x = np.linspace(-2.0, 2.0, 30)
    y = 1.5 * x**2 - 3.0 * x + 4.0
    model = PolynomialRegression(x, y, degree=2)
    result = model.fit()
    np.testing.assert_allclose(result.coefficients, [1.5, -3.0, 4.0], atol=1e-7)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)


def test_residuals_are_small_for_lightly_noisy_data():
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, 200)
    y_true = 2.0 * x**3 - x + 0.5
    y = y_true + rng.normal(0.0, 0.01, x.shape)
    model = PolynomialRegression(x, y, degree=3)
    result = model.fit()
    assert np.max(np.abs(result.residuals)) < 0.1
    assert result.r_squared > 0.99


def test_predict_matches_fitted_values_at_training_points():
    x = np.linspace(0.0, 5.0, 20)
    y = x**2 + 1.0
    model = PolynomialRegression(x, y, degree=2)
    result = model.fit()
    np.testing.assert_allclose(model.predict(x), result.fitted_values, atol=1e-8)


def test_condition_number_grows_with_degree():
    """Higher-degree Vandermonde fits on the same data are more
    ill-conditioned -- the classic motivation for preferring orthogonal
    polynomial bases at high degree."""
    x = np.linspace(-1.0, 1.0, 30)
    y = np.sin(3.0 * x)
    cond_low = PolynomialRegression(x, y, degree=2).fit().condition_number
    cond_high = PolynomialRegression(x, y, degree=8).fit().condition_number
    assert cond_high > cond_low


def test_rejects_degree_too_large_for_data():
    with pytest.raises(ValueError):
        PolynomialRegression([0.0, 1.0], [0.0, 1.0], degree=5)

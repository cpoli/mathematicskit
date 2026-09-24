"""Tests for Levenberg-Marquardt against exact fits and the linear least-squares solution."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.least_squares import levenberg_marquardt


def test_recovers_exponential_decay_parameters_from_exact_data():
    t = np.linspace(0.0, 5.0, 30)
    y = 3.0 * np.exp(-0.7 * t) + 0.5

    def residual(p):
        return p[0] * np.exp(-p[1] * t) + p[2] - y

    def jac(p):
        e = np.exp(-p[1] * t)
        return np.column_stack([e, -p[0] * t * e, np.ones_like(t)])

    result = levenberg_marquardt(residual, [1.0, 1.0, 0.0], jac=jac)
    assert result.success
    np.testing.assert_allclose(result.x, [3.0, 0.7, 0.5], atol=1e-8)
    assert result.cost == pytest.approx(0.0, abs=1e-16)


def test_linear_residual_matches_normal_equations():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 3))
    b = rng.normal(size=20)
    result = levenberg_marquardt(lambda x: a @ x - b, np.zeros(3))
    x_ls = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(result.x, x_ls, atol=1e-8)
    assert result.cost == pytest.approx(0.5 * np.sum((a @ x_ls - b) ** 2))
    np.testing.assert_allclose(result.residuals, a @ result.x - b)


def test_rosenbrock_as_least_squares():
    # Rosenbrock = r1^2 + r2^2 with r1 = 10(y - x^2), r2 = 1 - x.
    result = levenberg_marquardt(lambda p: np.array([10.0 * (p[1] - p[0] ** 2), 1.0 - p[0]]), [-1.2, 1.0])
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-8)

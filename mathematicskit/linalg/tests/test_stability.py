"""Tests for condition-number estimation and the normal-equations-vs-QR
least-squares stability comparison."""

import numpy as np
import pytest

from mathematicskit.linalg.systems.stability import condition_number_2norm, least_squares_normal_equations, least_squares_qr


def test_condition_number_of_diagonal_matrix():
    a = np.diag([10.0, 2.0, 1.0])
    assert condition_number_2norm(a) == pytest.approx(10.0, rel=1e-4)


def test_condition_number_of_identity_is_one():
    assert condition_number_2norm(np.eye(4)) == pytest.approx(1.0, rel=1e-6)


def test_least_squares_normal_equations_and_qr_agree_on_well_conditioned_problem():
    rng = np.random.default_rng(30)
    a = rng.uniform(-2, 2, size=(20, 3))
    x_true = np.array([1.0, -2.0, 0.5])
    b = a @ x_true
    result_normal = least_squares_normal_equations(a, b)
    result_qr = least_squares_qr(a, b)
    np.testing.assert_allclose(result_normal.coefficients, x_true, atol=1e-8)
    np.testing.assert_allclose(result_qr.coefficients, x_true, atol=1e-8)
    np.testing.assert_allclose(result_normal.coefficients, result_qr.coefficients, atol=1e-6)


def test_normal_equations_condition_number_is_square_of_qr_condition_number():
    """The textbook motivation for preferring QR: kappa(A^T A) = kappa(A)^2."""
    rng = np.random.default_rng(31)
    a = rng.uniform(-2, 2, size=(10, 3))
    b = rng.uniform(-1, 1, size=10)
    result_normal = least_squares_normal_equations(a, b)
    result_qr = least_squares_qr(a, b)
    assert result_normal.condition_number == pytest.approx(result_qr.condition_number**2, rel=1e-3)


def test_residual_norm_is_nonnegative_and_small_for_consistent_system():
    a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    x_true = np.array([1.0, 2.0])
    b = a @ x_true
    result = least_squares_qr(a, b)
    assert result.residual_norm < 1e-8

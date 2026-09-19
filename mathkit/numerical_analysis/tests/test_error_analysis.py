"""Tests for the shared error/stability-analysis utilities."""

import numpy as np
import pytest

from mathkit.numerical_analysis.utils.error_analysis import condition_number, estimate_convergence_order


def test_estimate_convergence_order_linear_sequence():
    errors = [0.5, 0.25, 0.125, 0.0625]  # ratio 0.5 each step -> order 1
    history = np.array([1.0 - e for e in errors])
    order = estimate_convergence_order(history, 1.0)
    assert order == pytest.approx(1.0, abs=1e-6)


def test_estimate_convergence_order_quadratic_sequence():
    errors = [0.1, 0.01, 0.0001, 1e-8]  # e_{k+1} = e_k^2 -> order 2
    history = np.array([1.0 - e for e in errors])
    order = estimate_convergence_order(history, 1.0)
    assert order == pytest.approx(2.0, abs=1e-6)


def test_condition_number_diagonal_matrix():
    a = np.diag([2.0, 5.0, 50.0])
    assert condition_number(a) == pytest.approx(25.0, rel=1e-6)


def test_condition_number_identity_is_one():
    assert condition_number(np.eye(4)) == pytest.approx(1.0, rel=1e-6)


def test_condition_number_agrees_with_numpy():
    rng = np.random.default_rng(2)
    a = rng.uniform(-3, 3, size=(5, 5))
    assert condition_number(a) == pytest.approx(np.linalg.cond(a), rel=1e-10)


def test_condition_number_rejects_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        condition_number(np.zeros((3, 3)))

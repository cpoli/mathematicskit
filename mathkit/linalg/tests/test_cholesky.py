"""Tests for Cholesky decomposition against closed-form/known results."""

import numpy as np
import pytest

from mathkit.linalg.systems.cholesky import cholesky_decompose, cholesky_solve, is_symmetric_positive_definite
from mathkit.linalg.utils.matrix_utils import random_spd_matrix


def test_cholesky_reconstructs_matrix():
    a = np.array([[4.0, 2.0], [2.0, 5.0]])
    result = cholesky_decompose(a)
    np.testing.assert_allclose(result.L @ result.L.T, a, atol=1e-10)


def test_cholesky_l_is_lower_triangular():
    a = random_spd_matrix(6, seed=7)
    result = cholesky_decompose(a)
    assert np.allclose(result.L, np.tril(result.L))


def test_cholesky_solve_matches_known_solution():
    a = np.array([[4.0, 2.0], [2.0, 5.0]])
    x_true = np.array([1.5, -0.5])
    b = a @ x_true
    x = cholesky_solve(cholesky_decompose(a), b)
    np.testing.assert_allclose(x, x_true, atol=1e-8)


def test_cholesky_rejects_non_spd():
    with pytest.raises(np.linalg.LinAlgError):
        cholesky_decompose(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError):
        cholesky_decompose(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_is_symmetric_positive_definite_flags_correctly():
    assert is_symmetric_positive_definite(np.array([[4.0, 2.0], [2.0, 5.0]]))
    assert not is_symmetric_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not is_symmetric_positive_definite(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_cholesky_agrees_with_lu_solve():
    from mathkit.linalg.systems.lu import lu_solve_system

    a = random_spd_matrix(5, seed=9)
    b = np.arange(1.0, 6.0)
    x_chol = cholesky_solve(cholesky_decompose(a), b)
    x_lu = lu_solve_system(a, b)
    np.testing.assert_allclose(x_chol, x_lu, atol=1e-7)

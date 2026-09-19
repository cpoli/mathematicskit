"""Tests for LU decomposition against closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.linalg.systems.lu import lu_decompose, lu_det, lu_solve_system
from mathematicskit.linalg.utils.matrix_utils import random_spd_matrix


def test_lu_reconstructs_matrix():
    a = np.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    result = lu_decompose(a)
    np.testing.assert_allclose(result.P @ a, result.L @ result.U, atol=1e-10)


def test_lu_l_is_unit_lower_triangular():
    a = random_spd_matrix(5, seed=3)
    result = lu_decompose(a)
    np.testing.assert_allclose(np.diag(result.L), np.ones(5))
    assert np.allclose(result.L, np.tril(result.L))


def test_lu_solve_matches_known_solution():
    a = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
    x_true = np.array([1.0, -2.0, -2.0])
    b = a @ x_true
    x = lu_solve_system(a, b)
    np.testing.assert_allclose(x, x_true, atol=1e-8)


def test_lu_det_matches_2x2_formula():
    a = np.array([[3.0, 8.0], [4.0, 6.0]])
    assert lu_det(a) == pytest.approx(3.0 * 6.0 - 8.0 * 4.0)


def test_lu_det_matches_numpy_on_random_matrices():
    rng = np.random.default_rng(4)
    for _ in range(5):
        a = rng.uniform(-5, 5, size=(4, 4))
        assert lu_det(a) == pytest.approx(np.linalg.det(a), rel=1e-6)


def test_lu_rejects_singular_matrix():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(np.linalg.LinAlgError):
        lu_decompose(a)

"""Tests for eigenvalue computation (numpy.linalg.eigh/eig) and the
hand-rolled power/inverse iteration, against closed-form/known
eigenvalues."""

import numpy as np
import pytest

from mathematicskit.linalg.systems.eigen import eigen_general, eigen_symmetric, inverse_iteration, power_iteration


def test_eigen_symmetric_matches_known_eigenvalues():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])  # eigenvalues 1, 3
    result = eigen_symmetric(a)
    assert result.converged
    np.testing.assert_allclose(sorted(result.eigenvalues), [1.0, 3.0], atol=1e-8)


def test_eigen_symmetric_eigenvectors_are_orthonormal_and_satisfy_av_eq_lambda_v():
    rng = np.random.default_rng(11)
    m = rng.uniform(-1, 1, size=(4, 4))
    a = m + m.T  # symmetric
    result = eigen_symmetric(a)
    np.testing.assert_allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(4), atol=1e-8)
    for i in range(4):
        lam, v = result.eigenvalues[i], result.eigenvectors[:, i]
        np.testing.assert_allclose(a @ v, lam * v, atol=1e-6)


def test_eigen_symmetric_rejects_non_symmetric():
    with pytest.raises(ValueError):
        eigen_symmetric(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_eigen_general_matches_known_complex_eigenvalues():
    a = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90-degree rotation: eigenvalues +-i
    result = eigen_general(a)
    np.testing.assert_allclose(sorted(result.eigenvalues.imag), [-1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(result.eigenvalues.real, [0.0, 0.0], atol=1e-8)


def test_eigen_general_agrees_with_eigen_symmetric_on_symmetric_input():
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    eig_general = sorted(eigen_general(a).eigenvalues.real)
    eig_symmetric = sorted(eigen_symmetric(a).eigenvalues)
    np.testing.assert_allclose(eig_general, eig_symmetric, atol=1e-8)


def test_power_iteration_finds_dominant_eigenvalue():
    a = np.diag([2.0, 5.0, -1.0])
    result = power_iteration(a)
    assert result.eigenvalues[0] == pytest.approx(5.0, abs=1e-6)


def test_inverse_iteration_finds_eigenvalue_nearest_shift():
    a = np.diag([2.0, 5.0, 9.0])
    result = inverse_iteration(a, mu=4.5)
    assert result.eigenvalues[0] == pytest.approx(5.0, abs=1e-6)
    result2 = inverse_iteration(a, mu=8.0)
    assert result2.eigenvalues[0] == pytest.approx(9.0, abs=1e-6)

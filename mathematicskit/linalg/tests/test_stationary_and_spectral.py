"""Tests for Jacobi/Gauss-Seidel/SOR, Lanczos (eigsh), and Hessenberg/Schur forms."""

import numpy as np
import pytest
import scipy.sparse as sp

from mathematicskit.linalg.systems.lanczos import lanczos_eigsh
from mathematicskit.linalg.systems.schur import hessenberg_reduce, schur_decompose
from mathematicskit.linalg.systems.stationary import SOR, GaussSeidel, JacobiIteration, jacobi_spectral_radius, optimal_sor_omega


def poisson_1d(n):
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


@pytest.mark.parametrize("solver", [JacobiIteration(tol=1e-12), GaussSeidel(tol=1e-12), SOR(omega=1.3, tol=1e-12)])
def test_stationary_solvers_converge_on_diagonally_dominant_system(solver):
    a = np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
    x_true = np.array([1.0, 2.0, -1.0])
    result = solver.solve(a, a @ x_true)
    assert result.converged
    np.testing.assert_allclose(result.x, x_true, atol=1e-10)
    assert result.residual_history.shape == (result.iterations + 1,)


def test_jacobi_spectral_radius_of_poisson_matrix_is_cos_pi_over_n_plus_1():
    n = 12
    assert jacobi_spectral_radius(poisson_1d(n)) == pytest.approx(np.cos(np.pi / (n + 1)), rel=1e-10)


def test_gauss_seidel_asymptotic_rate_is_jacobi_rate_squared():
    n = 10
    a = poisson_1d(n)
    b = np.ones(n)
    rho_j = np.cos(np.pi / (n + 1))
    rj = JacobiIteration(tol=1e-300, max_iter=400).solve(a, b).residual_history
    rgs = GaussSeidel(tol=1e-300, max_iter=200).solve(a, b).residual_history
    # asymptotic per-sweep contraction of the residual
    rate_j = (rj[300] / rj[200]) ** (1 / 100)
    rate_gs = (rgs[150] / rgs[100]) ** (1 / 50)
    assert rate_j == pytest.approx(rho_j, rel=1e-3)
    assert rate_gs == pytest.approx(rho_j**2, rel=1e-3)


def test_optimal_sor_omega_closed_form_and_speedup():
    n = 30
    a = poisson_1d(n)
    rho_j = np.cos(np.pi / (n + 1))
    omega = optimal_sor_omega(a)
    assert omega == pytest.approx(2.0 / (1.0 + np.sin(np.pi / (n + 1))), rel=1e-10)
    b = np.ones(n)
    it_gs = GaussSeidel(tol=1e-8, max_iter=10_000).solve(a, b).iterations
    it_sor = SOR(omega=omega, tol=1e-8, max_iter=10_000).solve(a, b).iterations
    assert rho_j < 1.0
    assert it_sor * 5 < it_gs


def test_sor_rejects_bad_omega_and_zero_diagonal():
    with pytest.raises(ValueError):
        SOR(omega=2.0)
    with pytest.raises(ValueError):
        GaussSeidel().solve(np.array([[0.0, 1.0], [1.0, 0.0]]), np.ones(2))


def test_lanczos_matches_closed_form_poisson_eigenvalues_on_sparse_matrix():
    n = 2000
    a = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    result = lanczos_eigsh(a, k=4, which="LA")
    j = np.arange(n - 3, n + 1)
    expected = 2.0 - 2.0 * np.cos(j * np.pi / (n + 1))
    np.testing.assert_allclose(result.eigenvalues, expected, rtol=1e-10)
    assert result.eigenvectors.shape == (n, 4)
    np.testing.assert_allclose(a @ result.eigenvectors, result.eigenvectors * result.eigenvalues, atol=1e-8)


def test_lanczos_agrees_with_eigh_on_dense_symmetric_matrix():
    rng = np.random.default_rng(50)
    m = rng.normal(size=(40, 40))
    a = m + m.T
    full = np.linalg.eigh(a)[0]
    np.testing.assert_allclose(lanczos_eigsh(a, k=3, which="SA").eigenvalues, full[:3], rtol=1e-9)


def test_hessenberg_reduction_is_similarity_with_zero_lower_part():
    a = np.random.default_rng(51).normal(size=(6, 6))
    h, q = hessenberg_reduce(a)
    np.testing.assert_allclose(np.tril(h, -2), 0.0, atol=1e-12)
    np.testing.assert_allclose(q @ h @ q.T, a, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(6), atol=1e-12)


def test_schur_of_rotation_has_complex_eigenvalues():
    theta = 0.7
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    result = schur_decompose(r, output="complex")
    np.testing.assert_allclose(np.sort(result.eigenvalues.imag), [-np.sin(theta), np.sin(theta)], atol=1e-12)
    np.testing.assert_allclose(result.Z @ result.T @ result.Z.conj().T, r, atol=1e-12)
    np.testing.assert_allclose(np.tril(result.T, -1), 0.0, atol=1e-12)


def test_schur_eigenvalues_match_triangular_matrix_diagonal():
    t = np.triu(np.arange(1.0, 17.0).reshape(4, 4))
    a = np.random.default_rng(52).normal(size=(4, 4))
    q, _ = np.linalg.qr(a)
    result = schur_decompose(q @ t @ q.T)
    np.testing.assert_allclose(np.sort(result.eigenvalues.real), np.sort(np.diag(t)), atol=1e-10)

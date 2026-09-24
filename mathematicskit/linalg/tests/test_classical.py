"""Tests for Cramer's rule, characteristic polynomials / Cayley-Hamilton,
the Moore-Penrose pseudoinverse, and Gershgorin discs."""

import numpy as np
import pytest

from mathematicskit.linalg.systems.cramer import cramer_solve
from mathematicskit.linalg.systems.gershgorin import gershgorin_discs
from mathematicskit.linalg.systems.matrix_polynomial import characteristic_polynomial, matrix_polynomial
from mathematicskit.linalg.systems.pseudoinverse import penrose_residuals, pseudoinverse


def test_cramer_solves_3x3_integer_system_exactly():
    # 2x + y - z = 8, -3x - y + 2z = -11, -2x + y + 2z = -3  ->  (2, 3, -1)
    a = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    np.testing.assert_allclose(cramer_solve(a, b), [2.0, 3.0, -1.0], atol=1e-12)


def test_cramer_matches_numpy_solve_on_random_system():
    rng = np.random.default_rng(40)
    a = rng.normal(size=(5, 5))
    b = rng.normal(size=5)
    np.testing.assert_allclose(cramer_solve(a, b), np.linalg.solve(a, b), rtol=1e-9)


def test_cramer_rejects_singular_matrix():
    with pytest.raises(np.linalg.LinAlgError):
        cramer_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


def test_characteristic_polynomial_trace_and_determinant():
    rng = np.random.default_rng(41)
    a = rng.normal(size=(4, 4))
    p = characteristic_polynomial(a)
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(-np.trace(a))
    assert p[-1] == pytest.approx(np.linalg.det(a))  # (-1)^4 det A


def test_characteristic_polynomial_of_companion_matrix_recovers_polynomial():
    # companion matrix of l^3 - 6 l^2 + 11 l - 6 = (l-1)(l-2)(l-3)
    c = np.array([[6.0, -11.0, 6.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(characteristic_polynomial(c), [1.0, -6.0, 11.0, -6.0], atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cayley_hamilton_theorem(seed):
    a = np.random.default_rng(seed).normal(size=(5, 5))
    residual = matrix_polynomial(characteristic_polynomial(a), a)
    assert np.linalg.norm(residual) < 1e-9 * np.linalg.norm(a) ** 5


def test_matrix_polynomial_matches_explicit_powers():
    a = np.array([[1.0, 2.0], [0.0, 3.0]])
    expected = 2 * a @ a - a + 4 * np.eye(2)
    np.testing.assert_allclose(matrix_polynomial([2.0, -1.0, 4.0], a), expected)


def test_pseudoinverse_equals_inverse_for_nonsingular_matrix():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(pseudoinverse(a), np.linalg.inv(a), atol=1e-12)


def test_pseudoinverse_of_rank_one_matrix_closed_form():
    # A = u v^T  ->  A^+ = v u^T / (|u|^2 |v|^2)
    u = np.array([1.0, 2.0, 2.0])
    v = np.array([3.0, 4.0])
    a = np.outer(u, v)
    np.testing.assert_allclose(pseudoinverse(a), np.outer(v, u) / (9.0 * 25.0), atol=1e-12)


def test_pseudoinverse_gives_minimum_norm_least_squares_solution():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(6, 3)) @ rng.normal(size=(3, 5))  # 6x5, rank 3
    b = rng.normal(size=6)
    x = pseudoinverse(a) @ b
    x_lstsq = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(x, x_lstsq, atol=1e-10)


def test_penrose_residuals_vanish_only_for_the_pseudoinverse():
    a = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
    assert np.all(penrose_residuals(a, pseudoinverse(a)) < 1e-12)
    assert np.max(penrose_residuals(a, pseudoinverse(a) + 0.01)) > 1e-3


def test_gershgorin_discs_contain_every_eigenvalue():
    rng = np.random.default_rng(43)
    for _ in range(20):
        a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        for by in ("rows", "columns"):
            discs = gershgorin_discs(a, by=by)
            assert all(discs.contains(lam) for lam in np.linalg.eigvals(a))


def test_gershgorin_radii_closed_form_and_diagonal_matrix():
    a = np.array([[5.0, -1.0, 2.0], [0.5, -3.0, 0.5], [0.0, 1.0, 1.0]])
    discs = gershgorin_discs(a)
    np.testing.assert_allclose(discs.centers, [5.0, -3.0, 1.0])
    np.testing.assert_allclose(discs.radii, [3.0, 1.0, 1.0])
    np.testing.assert_allclose(gershgorin_discs(a, by="columns").radii, [0.5, 2.0, 2.5])
    d = gershgorin_discs(np.diag([1.0, 2.0]))
    np.testing.assert_allclose(d.radii, 0.0)
    assert not d.contains(1.5)

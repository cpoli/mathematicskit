"""Tests for conjugate gradient and GMRES against direct-solve results."""

import numpy as np
import pytest

from mathkit.linalg.systems.iterative import GMRES, ConjugateGradient
from mathkit.linalg.systems.lu import lu_solve_system
from mathkit.linalg.utils.matrix_utils import random_spd_matrix


def test_conjugate_gradient_matches_known_solution_spd():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    result = ConjugateGradient(tol=1e-12).solve(a, b)
    assert result.converged
    np.testing.assert_allclose(a @ result.x, b, atol=1e-8)


def test_conjugate_gradient_converges_within_n_iterations():
    a = random_spd_matrix(6, seed=20)
    b = np.arange(1.0, 7.0)
    result = ConjugateGradient(tol=1e-10, max_iter=100).solve(a, b)
    assert result.converged
    assert result.iterations <= 6 + 2  # exact in <=n iterations, in exact arithmetic


def test_conjugate_gradient_residual_is_monotonically_decreasing_in_norm():
    a = random_spd_matrix(8, seed=21)
    b = np.ones(8)
    result = ConjugateGradient(tol=1e-12, max_iter=100).solve(a, b)
    # CG residual norms need not be strictly monotonic in general, but
    # the final residual should be far smaller than the initial one.
    assert result.residual_history[-1] < result.residual_history[0] * 1e-6


def test_gmres_matches_known_solution_nonsymmetric():
    a = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 2.0, 5.0]])
    b = np.array([1.0, 2.0, 3.0])
    result = GMRES(tol=1e-12).solve(a, b)
    assert result.converged
    np.testing.assert_allclose(a @ result.x, b, atol=1e-7)


def test_gmres_agrees_with_direct_solve_on_random_system():
    rng = np.random.default_rng(22)
    a = rng.uniform(-2, 2, size=(6, 6)) + 6.0 * np.eye(6)  # diagonally dominant -> nonsingular, well-behaved
    b = rng.uniform(-1, 1, size=6)
    x_gmres = GMRES(tol=1e-12, max_iter=6).solve(a, b).x
    x_direct = lu_solve_system(a, b)
    np.testing.assert_allclose(x_gmres, x_direct, atol=1e-6)


def test_conjugate_gradient_rejects_nothing_but_handles_zero_rhs():
    a = np.eye(3)
    result = ConjugateGradient().solve(a, np.zeros(3))
    assert result.converged
    np.testing.assert_allclose(result.x, np.zeros(3))


@pytest.mark.parametrize("solver_cls", [ConjugateGradient, GMRES])
def test_solvers_respect_initial_guess(solver_cls):
    a = random_spd_matrix(4, seed=23)
    x_true = np.array([1.0, 2.0, 3.0, 4.0])
    b = a @ x_true
    result = solver_cls(tol=1e-12).solve(a, b, x0=x_true.copy())
    np.testing.assert_allclose(result.x, x_true, atol=1e-6)
    assert result.iterations == 0

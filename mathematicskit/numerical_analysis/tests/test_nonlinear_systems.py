"""Tests for Newton's and Broyden's methods for nonlinear systems against
closed-form roots and scipy.optimize.root (Burden & Faires, Numerical
Analysis, 10th ed., Ch. 10)."""

import numpy as np
import pytest
from scipy.optimize import root

from mathematicskit.numerical_analysis import Broyden, NewtonSystem, numerical_jacobian


def _parabola_line(v):
    """y = x**2 meets x + y = 2 at (1, 1) and (-2, 4)."""
    return np.array([v[1] - v[0] ** 2, v[0] + v[1] - 2.0])


def _parabola_line_jacobian(v):
    return np.array([[-2.0 * v[0], 1.0], [1.0, 1.0]])


def _burden_faires_10_2(v):
    """Burden & Faires, sec. 10.2, Example 1: root (0.5, 0, -pi/6)."""
    x1, x2, x3 = v
    return np.array(
        [
            3.0 * x1 - np.cos(x2 * x3) - 0.5,
            x1**2 - 81.0 * (x2 + 0.1) ** 2 + np.sin(x3) + 1.06,
            np.exp(-x1 * x2) + 20.0 * x3 + (10.0 * np.pi - 3.0) / 3.0,
        ]
    )


def test_numerical_jacobian_matches_analytic():
    x0 = np.array([1.5, -0.5])
    np.testing.assert_allclose(numerical_jacobian(_parabola_line, x0), _parabola_line_jacobian(x0), atol=1e-8)


@pytest.mark.parametrize("x0,expected", [([0.5, 0.5], [1.0, 1.0]), ([-3.0, 5.0], [-2.0, 4.0])])
def test_newton_finds_both_intersections(x0, expected):
    result = NewtonSystem(_parabola_line, x0, jacobian=_parabola_line_jacobian, tol=1e-13).solve()
    assert result.converged
    np.testing.assert_allclose(result.root, expected, atol=1e-12)
    assert result.method == "newton"
    assert result.history.shape == (result.iterations + 1, 2)
    assert result.residual_norms.shape == (result.iterations + 1,)


def test_newton_matches_textbook_example_and_scipy():
    result = NewtonSystem(_burden_faires_10_2, [0.1, 0.1, -0.1], tol=1e-12).solve()
    assert result.converged
    np.testing.assert_allclose(result.root, [0.5, 0.0, -np.pi / 6], atol=1e-10)
    np.testing.assert_allclose(result.root, root(_burden_faires_10_2, [0.1, 0.1, -0.1], tol=1e-14).x, atol=1e-10)


def test_newton_converges_quadratically():
    """||F|| is roughly squared per step once close to the root."""
    norms = NewtonSystem(_parabola_line, [0.9, 1.3], jacobian=_parabola_line_jacobian, tol=1e-15).solve().residual_norms
    norms = norms[norms > 1e-13]
    ratios = np.log(norms[2:]) / np.log(norms[1:-1])
    assert ratios[-1] == pytest.approx(2.0, abs=0.3)


def test_newton_counts_finite_difference_evaluations():
    """With a 2 x 2 central-difference Jacobian each iteration costs 4 + 1 evaluations."""
    result = NewtonSystem(_parabola_line, [0.5, 0.5], tol=1e-12).solve()
    assert result.function_evaluations == 1 + 5 * result.iterations
    analytic = NewtonSystem(_parabola_line, [0.5, 0.5], jacobian=_parabola_line_jacobian, tol=1e-12).solve()
    assert analytic.function_evaluations == 1 + analytic.iterations


def test_newton_stops_on_singular_jacobian():
    """At x = -1/2 the parabola-line Jacobian [[1, 1], [1, 1]] is singular."""
    result = NewtonSystem(_parabola_line, [-0.5, 3.0], jacobian=_parabola_line_jacobian).solve()
    assert not result.converged
    assert result.iterations == 0
    assert result.extra["stopped"] == "singular Jacobian"
    np.testing.assert_array_equal(result.root, [-0.5, 3.0])


def test_newton_reports_max_iter_exhaustion():
    result = NewtonSystem(_burden_faires_10_2, [0.1, 0.1, -0.1], max_iter=2).solve()
    assert not result.converged
    assert result.iterations == 2


def test_broyden_matches_newton_root_with_fewer_evaluations():
    newton = NewtonSystem(_burden_faires_10_2, [0.1, 0.1, -0.1], tol=1e-12).solve()
    broyden = Broyden(_burden_faires_10_2, [0.1, 0.1, -0.1], tol=1e-12).solve()
    assert broyden.converged
    assert broyden.method == "broyden"
    np.testing.assert_allclose(broyden.root, newton.root, atol=1e-10)
    np.testing.assert_allclose(broyden.root, root(_burden_faires_10_2, [0.1, 0.1, -0.1], method="broyden1", tol=1e-12).x, atol=1e-8)
    assert broyden.iterations > newton.iterations
    assert broyden.function_evaluations == 1 + 6 + broyden.iterations
    assert broyden.function_evaluations < newton.function_evaluations


def test_broyden_update_satisfies_the_secant_condition():
    """After each step, B_{k+1} s_k = F(x_{k+1}) - F(x_k); for a linear map
    the final B therefore acts like the true matrix on the last step."""
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    F = lambda x: A @ x - b
    result = Broyden(F, np.zeros(3), jacobian=lambda x: np.eye(3), tol=1e-12, max_iter=50).solve()
    assert result.converged
    np.testing.assert_allclose(result.root, np.linalg.solve(A, b), atol=1e-10)
    s = result.history[-1] - result.history[-2]
    B = result.extra["jacobian_approximation"]
    np.testing.assert_allclose(B @ s, A @ s, atol=1e-12)


def test_broyden_converges_superlinearly():
    norms = Broyden(_parabola_line, [0.9, 1.3], tol=1e-15).solve().residual_norms
    norms = norms[norms > 1e-14]
    ratios = norms[1:] / norms[:-1]
    assert ratios[-1] < 0.1 * ratios[1]


def test_broyden_stops_on_singular_initial_jacobian():
    result = Broyden(_parabola_line, [-0.5, 3.0], jacobian=_parabola_line_jacobian).solve()
    assert not result.converged
    assert result.iterations == 0
    assert result.extra["stopped"] == "singular Jacobian approximation"

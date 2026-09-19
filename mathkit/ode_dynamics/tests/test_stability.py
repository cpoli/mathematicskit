"""Tests for 2D fixed-point stability classification against textbook
closed-form cases (Strogatz, Nonlinear Dynamics and Chaos, Ch. 5)."""

import numpy as np
import pytest

from mathkit.ode_dynamics.systems.stability import classify_fixed_point_2d, find_fixed_point_newton, numerical_jacobian


def test_stable_node():
    result = classify_fixed_point_2d(np.array([[-1.0, 0.0], [0.0, -2.0]]))
    assert result.classification == "stable node"
    assert result.stable


def test_unstable_node():
    result = classify_fixed_point_2d(np.array([[1.0, 0.0], [0.0, 2.0]]))
    assert result.classification == "unstable node"
    assert not result.stable


def test_saddle():
    result = classify_fixed_point_2d(np.array([[1.0, 0.0], [0.0, -1.0]]))
    assert result.classification == "saddle"
    assert not result.stable


def test_stable_spiral():
    result = classify_fixed_point_2d(np.array([[-0.1, 1.0], [-1.0, -0.1]]))
    assert result.classification == "stable spiral"
    assert result.stable


def test_unstable_spiral():
    result = classify_fixed_point_2d(np.array([[0.1, 1.0], [-1.0, 0.1]]))
    assert result.classification == "unstable spiral"
    assert not result.stable


def test_center():
    result = classify_fixed_point_2d(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    assert result.classification == "center"
    # A center is marginally stable (purely imaginary eigenvalues -> real part 0, not < 0).
    assert not result.stable


def test_numerical_jacobian_matches_known_linear_system():
    f = lambda x: np.array([2.0 * x[0] + 3.0 * x[1], -x[0] + x[1] ** 2])
    x0 = np.array([1.0, 2.0])
    jac = numerical_jacobian(f, x0)
    expected = np.array([[2.0, 3.0], [-1.0, 2.0 * x0[1]]])
    np.testing.assert_allclose(jac, expected, atol=1e-4)


def test_find_fixed_point_newton_matches_known_intersection():
    """Fixed point of the classic parabola-line intersection: y=x^2, x+y=2 -> x=1, y=1."""
    f = lambda x: np.array([x[1] - x[0] ** 2, x[0] + x[1] - 2.0])
    x, converged = find_fixed_point_newton(f, np.array([0.5, 0.5]))
    assert converged
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-6)


def test_classification_matches_jacobian_at_newton_solved_fixed_point():
    """Van der Pol's origin (mu>0) is an unstable spiral for small mu."""
    mu = 0.5
    f = lambda x: np.array([x[1], mu * (1.0 - x[0] ** 2) * x[1] - x[0]])
    x0, converged = find_fixed_point_newton(f, np.array([0.1, 0.1]))
    assert converged
    np.testing.assert_allclose(x0, [0.0, 0.0], atol=1e-6)
    jac = numerical_jacobian(f, x0)
    result = classify_fixed_point_2d(jac, location=x0)
    assert result.classification == "unstable spiral"


@pytest.mark.parametrize(
    "jac,expected",
    [
        (np.array([[-2.0, 0.0], [0.0, -2.0]]), "degenerate node"),
    ],
)
def test_degenerate_node(jac, expected):
    result = classify_fixed_point_2d(jac)
    assert result.classification == expected

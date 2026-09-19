"""Tests for Lagrange-multiplier stationary points, KKT verification, and
the penalty method against closed-form constrained-optimization results."""

import numpy as np
import pytest

from mathkit.optimization.systems.constrained import PenaltyMethod, lagrange_stationary_point, verify_kkt


def _circle_problem():
    """minimize x^2 + y^2 s.t. x + y = 1: exact solution (0.5, 0.5), lambda = -1."""
    f = lambda z: z[0] ** 2 + z[1] ** 2
    grad_f = lambda z: np.array([2.0 * z[0], 2.0 * z[1]])
    h = lambda z: np.array([z[0] + z[1] - 1.0])
    grad_h = lambda z: np.array([[1.0, 1.0]])
    return f, grad_f, h, grad_h


def test_lagrange_stationary_point_matches_closed_form():
    _, grad_f, h, grad_h = _circle_problem()
    result = lagrange_stationary_point(grad_f, h, grad_h, x0=np.array([0.0, 0.0]))
    assert result.converged
    np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-8)
    np.testing.assert_allclose(result.multipliers, [-1.0], atol=1e-8)


def test_verify_kkt_true_at_exact_stationary_point():
    _, grad_f, h, grad_h = _circle_problem()
    result = verify_kkt(np.array([0.5, 0.5]), grad_f, h=h, grad_h=grad_h, eq_multipliers=np.array([-1.0]))
    assert result.satisfied
    assert result.stationarity_residual < 1e-8


def test_verify_kkt_false_away_from_stationary_point():
    _, grad_f, h, grad_h = _circle_problem()
    result = verify_kkt(np.array([0.0, 1.0]), grad_f, h=h, grad_h=grad_h, eq_multipliers=np.array([-1.0]))
    assert not result.satisfied


def test_verify_kkt_inequality_constraint_dual_feasibility():
    """minimize x^2 s.t. x >= 1 (i.e. g(x) = 1 - x <= 0): stationary at
    x=1 with mu=2 >= 0 (dual-feasible, complementary since g(1)=0)."""
    grad_f = lambda z: np.array([2.0 * z[0]])
    g = lambda z: np.array([1.0 - z[0]])
    grad_g = lambda z: np.array([[-1.0]])
    result = verify_kkt(np.array([1.0]), grad_f, g=g, grad_g=grad_g, ineq_multipliers=np.array([2.0]))
    assert result.satisfied
    assert result.dual_feasible
    assert result.complementary_slackness

    bad = verify_kkt(np.array([1.0]), grad_f, g=g, grad_g=grad_g, ineq_multipliers=np.array([-2.0]))
    assert not bad.dual_feasible
    assert not bad.satisfied


def test_penalty_method_converges_to_closed_form_constrained_minimum():
    f, grad_f, h, grad_h = _circle_problem()
    result = PenaltyMethod(mu0=1.0, mu_factor=10.0, n_outer=8).minimize(f, grad_f, np.array([0.0, 0.0]), h=h, grad_h=grad_h)
    np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-3)


def test_penalty_method_records_mu_history():
    f, grad_f, h, grad_h = _circle_problem()
    result = PenaltyMethod(mu0=1.0, mu_factor=5.0, n_outer=4).minimize(f, grad_f, np.array([0.0, 0.0]), h=h, grad_h=grad_h)
    mu_history = result.extra["mu_history"]
    assert mu_history.shape[0] == 4
    np.testing.assert_allclose(mu_history, [1.0, 5.0, 25.0, 125.0])


def test_penalty_method_handles_inequality_constraint():
    """minimize x^2 s.t. x >= 1: penalty solution should approach x=1."""
    f = lambda z: z[0] ** 2
    grad_f = lambda z: np.array([2.0 * z[0]])
    g = lambda z: np.array([1.0 - z[0]])
    grad_g = lambda z: np.array([[-1.0]])
    result = PenaltyMethod(mu0=1.0, mu_factor=10.0, n_outer=8).minimize(f, grad_f, np.array([3.0]), g=g, grad_g=grad_g)
    assert result.x[0] == pytest.approx(1.0, abs=1e-2)

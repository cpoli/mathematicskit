"""Tests for linear programming against closed-form/known optimal solutions."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.linear_programming import linear_program


def test_simple_lp_matches_hand_solved_vertex():
    # minimize -x - 2y s.t. x + y <= 4, x <= 3, x, y >= 0.
    # Feasible vertices: (0,0),(3,0),(3,1),(0,4). Objective -x-2y minimized at (0,4)=-8.
    result = linear_program(c=np.array([-1.0, -2.0]), a_ub=np.array([[1.0, 1.0], [1.0, 0.0]]), b_ub=np.array([4.0, 3.0]))
    assert result.success
    np.testing.assert_allclose(result.x, [0.0, 4.0], atol=1e-6)
    assert result.fun == pytest.approx(-8.0, abs=1e-6)


def test_lp_with_equality_constraint():
    # minimize x + y s.t. x + 2y = 4, x, y >= 0 -> minimized along the
    # constraint boundary at y=2, x=0 (since y has a bigger coefficient
    # per unit of the constraint, pushing all weight to y minimizes x+y).
    result = linear_program(c=np.array([1.0, 1.0]), a_eq=np.array([[1.0, 2.0]]), b_eq=np.array([4.0]))
    assert result.success
    np.testing.assert_allclose(result.x, [0.0, 2.0], atol=1e-6)
    assert result.fun == pytest.approx(2.0, abs=1e-6)


def test_lp_with_custom_bounds():
    result = linear_program(c=np.array([1.0]), bounds=[(2.0, 5.0)])
    assert result.success
    np.testing.assert_allclose(result.x, [2.0], atol=1e-8)


def test_infeasible_lp_reports_failure():
    # x >= 5 and x <= 1 simultaneously is infeasible.
    result = linear_program(c=np.array([1.0]), a_ub=np.array([[1.0]]), b_ub=np.array([1.0]), bounds=[(5.0, None)])
    assert not result.success

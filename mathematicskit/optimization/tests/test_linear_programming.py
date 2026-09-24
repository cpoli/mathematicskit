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


def test_integer_program_differs_from_lp_relaxation():
    # maximize 5x + 4y s.t. 6x + 4y <= 24, x + 2y <= 6: LP optimum (3, 1.5) = 21,
    # integer optimum (4, 0) = 20 (checked by enumerating the feasible lattice points).
    from mathematicskit.optimization.systems.linear_programming import integer_linear_program

    c = np.array([-5.0, -4.0])
    a_ub = np.array([[6.0, 4.0], [1.0, 2.0]])
    b_ub = np.array([24.0, 6.0])
    relaxed = linear_program(c, a_ub=a_ub, b_ub=b_ub)
    np.testing.assert_allclose(relaxed.x, [3.0, 1.5], atol=1e-9)
    result = integer_linear_program(c, a_ub=a_ub, b_ub=b_ub)
    assert result.success
    np.testing.assert_allclose(result.x, [4.0, 0.0], atol=1e-9)
    best = max(5 * x + 4 * y for x in range(5) for y in range(4) if 6 * x + 4 * y <= 24 and x + 2 * y <= 6)
    assert -result.fun == pytest.approx(best)


def test_mixed_integer_program_with_equality_and_bounds():
    from mathematicskit.optimization.systems.linear_programming import integer_linear_program

    # minimize x + y, x + y = 2.5, x integer in [0, 1], y continuous: x = 0 or 1, total 2.5 either way;
    # add a tie-breaker so x = 1 is optimal.
    result = integer_linear_program(np.array([0.9, 1.0]), a_eq=np.array([[1.0, 1.0]]), b_eq=np.array([2.5]), bounds=[(0, 1), (0, None)], integrality=[1, 0])
    np.testing.assert_allclose(result.x, [1.0, 1.5], atol=1e-9)


def test_infeasible_integer_program_reports_failure():
    from mathematicskit.optimization.systems.linear_programming import integer_linear_program

    # 2x = 1 has no integer solution.
    result = integer_linear_program(np.array([1.0]), a_eq=np.array([[2.0]]), b_eq=np.array([1.0]))
    assert not result.success

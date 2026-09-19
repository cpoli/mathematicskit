"""Tests for linear and Pell Diophantine solvers against closed-form/known results."""

import pytest

from mathematicskit.number_theory.systems.diophantine import solve_linear_diophantine, solve_pell_equation


def test_linear_diophantine_solution_satisfies_equation():
    result = solve_linear_diophantine(3, 5, 1)
    assert result.has_solution
    assert 3 * result.x0 + 5 * result.y0 == 1


def test_linear_diophantine_general_solution_family():
    result = solve_linear_diophantine(6, 9, 3)
    assert result.has_solution
    for k in range(-5, 6):
        x = result.x0 + k * result.x_step
        y = result.y0 - k * result.y_step
        assert 6 * x + 9 * y == 3


def test_linear_diophantine_no_solution_when_gcd_does_not_divide_c():
    result = solve_linear_diophantine(4, 6, 5)
    assert not result.has_solution
    assert result.gcd == 2


@pytest.mark.parametrize("d,expected", [(2, (3, 2)), (3, (2, 1)), (5, (9, 4)), (7, (8, 3))])
def test_pell_fundamental_solutions_match_known_values(d, expected):
    result = solve_pell_equation(d)
    assert (result.x, result.y) == expected


def test_pell_solution_satisfies_equation_for_larger_d():
    for d in (13, 19, 31, 61):
        result = solve_pell_equation(d)
        assert result.x**2 - d * result.y**2 == 1


def test_pell_rejects_perfect_square():
    with pytest.raises(ValueError):
        solve_pell_equation(16)

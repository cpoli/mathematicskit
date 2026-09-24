"""Tests for zero-sum matrix games against closed-form 2x2 solutions and the minimax theorem."""

import numpy as np
import pytest

from mathematicskit.optimization.systems.game_theory import solve_zero_sum_game


def test_rock_paper_scissors_is_uniform_with_value_zero():
    result = solve_zero_sum_game([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    np.testing.assert_allclose(result.row_strategy, [1 / 3] * 3, atol=1e-9)
    np.testing.assert_allclose(result.col_strategy, [1 / 3] * 3, atol=1e-9)
    assert result.value == pytest.approx(0.0, abs=1e-9)


def test_2x2_game_without_saddle_matches_closed_form():
    # For [[a, b], [c, d]] with no saddle point, p1 = (d - c)/(a - b - c + d),
    # q1 = (d - b)/(a - b - c + d), value = (ad - bc)/(a - b - c + d).
    a, b, c, d = 3.0, -1.0, -2.0, 4.0
    den = a - b - c + d
    result = solve_zero_sum_game([[a, b], [c, d]])
    np.testing.assert_allclose(result.row_strategy, [(d - c) / den, (a - b) / den], atol=1e-9)
    np.testing.assert_allclose(result.col_strategy, [(d - b) / den, (a - c) / den], atol=1e-9)
    assert result.value == pytest.approx((a * d - b * c) / den, abs=1e-9)


def test_saddle_point_gives_pure_strategies():
    # Row 1 dominates; column 1's minimum there (2) is the saddle point.
    result = solve_zero_sum_game([[1.0, 5.0], [2.0, 3.0]])
    assert result.value == pytest.approx(2.0, abs=1e-9)
    np.testing.assert_allclose(result.row_strategy, [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.col_strategy, [1.0, 0.0], atol=1e-9)


def test_minimax_equality_on_random_game():
    a = np.random.default_rng(3).normal(size=(4, 6))
    result = solve_zero_sum_game(a)
    p, q = result.row_strategy, result.col_strategy
    # p guarantees at least v against every column; q concedes at most v to every row.
    assert np.min(p @ a) == pytest.approx(result.value, abs=1e-8)
    assert np.max(a @ q) == pytest.approx(result.value, abs=1e-8)
    assert p @ a @ q == pytest.approx(result.value, abs=1e-8)

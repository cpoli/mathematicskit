"""Tests for elementary 1D cellular automata (Wolfram rules) and Conway's
Game of Life against known closed-form behavior: rule 90's XOR pattern,
rule 0/255's trivial fixed points, and Life's classic still lifes and
oscillators."""

import numpy as np
import pytest

from mathematicskit.fractals_chaos.systems.cellular_automata import ElementaryCA, GameOfLife


def test_rule_90_is_xor_of_neighbors():
    ca = ElementaryCA(rule=90, width=9)
    for _ in range(3):
        state = ca.state.copy()
        new_state = ca.step()
        left = np.roll(state, 1)
        right = np.roll(state, -1)
        np.testing.assert_array_equal(new_state, left ^ right)


def test_rule_0_kills_everything():
    ca = ElementaryCA(rule=0, width=11)
    ca.step()
    assert np.all(ca.state == 0)


def test_rule_255_fills_everything():
    ca = ElementaryCA(rule=255, width=11)
    ca.step()
    assert np.all(ca.state == 1)


def test_elementary_ca_run_records_every_generation():
    ca = ElementaryCA(rule=90, width=7)
    history = ca.run(5)
    assert history.shape == (6, 7)
    np.testing.assert_array_equal(history[0], [0, 0, 0, 1, 0, 0, 0])


def test_elementary_ca_rejects_invalid_rule():
    with pytest.raises(ValueError):
        ElementaryCA(rule=256, width=5)
    with pytest.raises(ValueError):
        ElementaryCA(rule=-1, width=5)


def test_game_of_life_block_is_a_still_life():
    grid = np.zeros((6, 6), dtype=np.int64)
    grid[2:4, 2:4] = 1  # 2x2 block
    life = GameOfLife(grid)
    next_state = life.step()
    np.testing.assert_array_equal(next_state, grid)


def test_game_of_life_blinker_has_period_two():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[2, 1:4] = 1
    life = GameOfLife(grid)
    after_one = life.step().copy()
    assert not np.array_equal(after_one, grid)
    after_two = life.step()
    np.testing.assert_array_equal(after_two, grid)


def test_game_of_life_empty_grid_stays_empty():
    grid = np.zeros((4, 4), dtype=np.int64)
    life = GameOfLife(grid)
    np.testing.assert_array_equal(life.step(), grid)


def test_game_of_life_rejects_non_2d_input():
    with pytest.raises(ValueError):
        GameOfLife(np.zeros(5))

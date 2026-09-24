"""Tests for Lindenmayer rewriting and turtle graphics."""

import numpy as np
import pytest

from mathematicskit.fractals_chaos.systems.curves import koch_curve
from mathematicskit.fractals_chaos.systems.lsystems import lsystem, turtle_path


def test_algae_lengths_are_fibonacci():
    lengths = [len(lsystem("A", {"A": "AB", "B": "A"}, n)) for n in range(10)]
    assert lengths == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def test_koch_lsystem_matches_direct_construction():
    word = lsystem("F", {"F": "F+F--F+F"}, 3)
    (line,) = turtle_path(word, 60, step=3.0**-3, heading=0)
    assert np.allclose(line, koch_curve(3))


def test_brackets_start_new_polylines():
    lines = turtle_path("F[+F]F", 90)
    assert len(lines) == 2
    assert np.allclose(lines[1][0], [0.0, 1.0])


def test_pen_up_move_does_not_draw():
    lines = turtle_path("FfF", 90, heading=0)
    assert [len(line) for line in lines] == [2, 2]
    assert np.allclose(lines[1][0], [2.0, 0.0])


def test_unbalanced_pop_raises():
    with pytest.raises(IndexError):
        turtle_path("F]", 90)

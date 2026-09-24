"""Tests for Latin squares and orthogonal pairs."""

from itertools import permutations

import numpy as np
import pytest

from mathematicskit.combinatorics.systems.designs import are_orthogonal, cyclic_latin_square, is_latin_square, orthogonal_latin_square_pair


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_cyclic_square_is_latin(n):
    assert is_latin_square(cyclic_latin_square(n))


def test_non_coprime_multiplier_rejected():
    with pytest.raises(ValueError):
        cyclic_latin_square(6, 2)


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_odd_order_pair_is_orthogonal(n):
    a, b = orthogonal_latin_square_pair(n)
    assert is_latin_square(a) and is_latin_square(b)
    assert are_orthogonal(a, b)


def test_no_orthogonal_mate_for_order_two():
    squares = [np.array(p) for p in ([[0, 1], [1, 0]], [[1, 0], [0, 1]])]
    assert not any(are_orthogonal(a, b) for a in squares for b in squares)


def test_order_three_latin_squares_count():
    rows = list(permutations(range(3)))
    count = sum(is_latin_square(np.array(sq)) for sq in permutations(rows, 3))
    assert count == 12

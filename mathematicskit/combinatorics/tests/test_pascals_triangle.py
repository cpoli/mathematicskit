"""Tests for the hand-rolled Pascal's triangle against
scipy.special.comb-computed binomial coefficients."""

import pytest

from mathematicskit.combinatorics.systems.counting import combinations_count
from mathematicskit.combinatorics.systems.pascals_triangle import pascals_triangle


def test_pascals_triangle_matches_binomial_coefficients():
    triangle = pascals_triangle(10)
    for n, row in enumerate(triangle):
        for k, value in enumerate(row):
            assert value == combinations_count(n, k)


def test_row_lengths_increase():
    triangle = pascals_triangle(6)
    assert [len(row) for row in triangle] == [1, 2, 3, 4, 5, 6]


def test_rejects_non_positive_n_rows():
    with pytest.raises(ValueError):
        pascals_triangle(0)

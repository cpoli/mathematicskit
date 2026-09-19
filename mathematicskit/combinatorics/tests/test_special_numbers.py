"""Tests for Stirling and Catalan numbers against closed-form/known
results and cross-checks against scipy-computed binomial coefficients."""

import math

import pytest

from mathematicskit.combinatorics.systems.counting import combinations_count
from mathematicskit.combinatorics.systems.special_numbers import catalan_number, stirling_first_kind, stirling_second_kind
from mathematicskit.combinatorics.utils.bell_number import bell_number


def test_stirling_first_kind_row_sums_to_factorial():
    for n in range(1, 8):
        assert sum(stirling_first_kind(n, k) for k in range(n + 1)) == math.factorial(n)


def test_stirling_first_kind_known_value():
    assert stirling_first_kind(4, 2) == 11


def test_stirling_first_kind_signed_alternates():
    for n in range(1, 6):
        for k in range(n + 1):
            unsigned = stirling_first_kind(n, k)
            signed = stirling_first_kind(n, k, signed=True)
            assert signed == (-1) ** (n - k) * unsigned


def test_stirling_second_kind_boundary_cases():
    for n in range(1, 8):
        assert stirling_second_kind(n, 1) == 1
        assert stirling_second_kind(n, n) == 1


def test_stirling_second_kind_known_value():
    assert stirling_second_kind(4, 2) == 7


def test_catalan_numbers_known_sequence():
    assert [catalan_number(n) for n in range(6)] == [1, 1, 2, 5, 14, 42]


def test_catalan_matches_closed_form():
    for n in range(15):
        expected = combinations_count(2 * n, n) // (n + 1)
        assert catalan_number(n) == expected


def test_bell_numbers_known_sequence():
    assert [bell_number(n) for n in range(6)] == [1, 1, 2, 5, 15, 52]


@pytest.mark.parametrize("n,k", [(-1, 2), (3, -1), (3, 5)])
def test_stirling_out_of_range_is_zero(n, k):
    assert stirling_second_kind(n, k) == 0

"""Tests for permutation/combination counting and generation against
closed-form/known results."""

import math

import pytest

from mathematicskit.combinatorics.systems.counting import (
    combinations_count,
    generate_combinations,
    generate_permutations,
    multinomial_coefficient,
    permutations_count,
)


def test_permutations_count_matches_factorial_ratio():
    assert permutations_count(5, 2) == 20
    assert permutations_count(5) == math.factorial(5)


def test_combinations_count_matches_math_comb():
    for n, k in [(5, 2), (52, 5), (10, 0), (10, 10)]:
        assert combinations_count(n, k) == math.comb(n, k)


def test_combinations_count_symmetry():
    assert combinations_count(10, 3) == combinations_count(10, 7)


def test_multinomial_coefficient_matches_factorial_definition():
    n, ks = 10, [2, 3, 5]
    expected = math.factorial(n) // (math.factorial(2) * math.factorial(3) * math.factorial(5))
    assert multinomial_coefficient(n, ks) == expected


def test_multinomial_reduces_to_binomial_for_two_parts():
    assert multinomial_coefficient(10, [3, 7]) == combinations_count(10, 3)


def test_multinomial_rejects_mismatched_sum():
    with pytest.raises(ValueError):
        multinomial_coefficient(10, [2, 3])


def test_generate_permutations_count_and_uniqueness():
    perms = generate_permutations([1, 2, 3, 4], r=2)
    assert len(perms) == permutations_count(4, 2)
    assert len(set(perms)) == len(perms)


def test_generate_combinations_count_and_uniqueness():
    combos = generate_combinations([1, 2, 3, 4, 5], r=3)
    assert len(combos) == combinations_count(5, 3)
    assert len(set(combos)) == len(combos)
    assert all(tuple(sorted(c)) == c for c in combos)

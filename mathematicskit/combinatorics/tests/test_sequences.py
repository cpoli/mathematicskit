"""Tests for Fibonacci numbers, Bernoulli numbers, power sums, and Gray codes."""

from fractions import Fraction

import pytest
from scipy import special

from mathematicskit.combinatorics.systems.sequences import bernoulli_numbers, domino_tilings, fibonacci, gray_code, sum_of_powers


def test_fibonacci_satisfies_cassini_identity():
    for n in range(1, 30):
        assert fibonacci(n + 1) * fibonacci(n - 1) - fibonacci(n) ** 2 == (-1) ** n


def test_fibonacci_matches_binet_formula():
    phi = (1 + 5**0.5) / 2
    for n in range(40):
        assert fibonacci(n) == round(phi**n / 5**0.5)


def test_domino_tilings_match_brute_force_compositions():
    def compositions(n):
        return 1 if n <= 1 else compositions(n - 1) + compositions(n - 2)

    for n in range(12):
        assert domino_tilings(n) == compositions(n)


def test_bernoulli_numbers_match_scipy_up_to_b1_sign():
    exact = bernoulli_numbers(20)
    floats = special.bernoulli(20)
    assert exact[1] == Fraction(1, 2)
    for m, value in enumerate(exact):
        if m != 1:
            assert float(value) == pytest.approx(floats[m], rel=1e-9, abs=1e-15)  # scipy returns floats, not exact values


@pytest.mark.parametrize("p", range(8))
def test_sum_of_powers_matches_direct_sum(p):
    for n in (0, 1, 7, 50):
        assert sum_of_powers(n, p) == sum(k**p for k in range(1, n + 1))


def test_gray_code_neighbours_differ_in_one_bit_and_cover_all_words():
    for n in range(1, 9):
        codes = gray_code(n)
        assert sorted(codes) == list(range(2**n))
        for a, b in zip(codes, codes[1:] + codes[:1]):
            assert bin(a ^ b).count("1") == 1

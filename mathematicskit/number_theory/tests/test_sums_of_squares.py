"""Tests for sums of two and four squares against Fermat's and Lagrange's
theorems."""

import pytest

from mathematicskit.number_theory.systems.primality import sieve_of_eratosthenes
from mathematicskit.number_theory.systems.sums_of_squares import sum_of_four_squares, sum_of_two_squares
from mathematicskit.number_theory.systems.totient import prime_factorization


def test_fermat_two_squares_for_odd_primes():
    for p in sieve_of_eratosthenes(2000)[1:]:
        p = int(p)
        rep = sum_of_two_squares(p)
        if p % 4 == 1:
            assert rep is not None
            a, b = rep
            assert a * a + b * b == p
        else:
            assert rep is None


def test_two_squares_criterion_for_composites():
    # n is a sum of two squares iff every prime 3 (mod 4) appears to an even power.
    for n in range(1, 1500):
        expected = all(e % 2 == 0 for q, e in prime_factorization(n).items() if q % 4 == 3)
        assert (sum_of_two_squares(n) is not None) == expected


def test_two_squares_known_values():
    assert sum_of_two_squares(0) == (0, 0)
    assert sum_of_two_squares(25) == (0, 5)
    assert sum_of_two_squares(50) == (1, 7)


def test_lagrange_four_squares_for_all_n():
    for n in range(0, 3000):
        a, b, c, d = sum_of_four_squares(n)
        assert a * a + b * b + c * c + d * d == n
        assert a >= b >= c >= d >= 0


def test_numbers_of_form_4k_times_8m_plus_7_need_four_nonzero_squares():
    # Legendre's three-square theorem: 4^k (8m + 7) is not a sum of three squares.
    for n in (7, 15, 28, 60, 112):
        assert min(sum_of_four_squares(n)) > 0


def test_negative_input_raises():
    with pytest.raises(ValueError):
        sum_of_two_squares(-1)
    with pytest.raises(ValueError):
        sum_of_four_squares(-1)

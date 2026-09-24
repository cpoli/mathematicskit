"""Tests for Pollard's rho factorization and the Lucas-Lehmer test."""

import pytest

from mathematicskit.number_theory.systems.factorization import pollard_rho
from mathematicskit.number_theory.systems.primality import is_prime_miller_rabin, lucas_lehmer


@pytest.mark.parametrize(
    ("n", "factors"),
    [
        (8051, {83, 97}),
        (10403, {101, 103}),
        (2**64 + 1, {274177, 67280421310721}),  # the Fermat number F_6
        (1000000007 * 998244353, {1000000007, 998244353}),
    ],
)
def test_pollard_rho_finds_known_factors(n, factors):
    result = pollard_rho(n)
    assert result.factor * result.cofactor == n
    assert {result.factor, result.cofactor} == factors
    assert result.iterations > 0


def test_pollard_rho_on_even_and_small_composites():
    assert pollard_rho(10).factor == 2
    for n in (9, 15, 21, 25, 49, 77, 91, 121, 143, 169):
        result = pollard_rho(n)
        assert 1 < result.factor < n and n % result.factor == 0


def test_pollard_rho_rejects_small_n():
    with pytest.raises(ValueError):
        pollard_rho(3)


def test_lucas_lehmer_mersenne_exponents():
    known = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607]
    assert [p for p in range(2, 650) if lucas_lehmer(p)] == known


def test_lucas_lehmer_agrees_with_miller_rabin():
    for p in range(2, 200):
        assert lucas_lehmer(p) == is_prime_miller_rabin(2**p - 1)


def test_lucas_lehmer_rejects_m11_despite_prime_exponent():
    assert not lucas_lehmer(11)  # 2^11 - 1 = 23 * 89

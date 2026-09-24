"""Tests for Legendre/Jacobi symbols and Tonelli-Shanks against brute force
and the law of quadratic reciprocity."""

import pytest

from mathematicskit.number_theory.systems.primality import sieve_of_eratosthenes
from mathematicskit.number_theory.systems.quadratic_residues import jacobi_symbol, legendre_symbol, sqrt_mod
from mathematicskit.number_theory.systems.totient import prime_factorization

ODD_PRIMES = [int(p) for p in sieve_of_eratosthenes(200)[1:]]


def test_legendre_symbol_matches_brute_force_squares():
    for p in ODD_PRIMES[:15]:
        squares = {x * x % p for x in range(1, p)}
        for a in range(p):
            expected = 0 if a == 0 else (1 if a in squares else -1)
            assert legendre_symbol(a, p) == expected


def test_quadratic_reciprocity():
    for p in ODD_PRIMES:
        for q in ODD_PRIMES:
            if p != q:
                sign = -1 if (p % 4 == 3 and q % 4 == 3) else 1
                assert legendre_symbol(p, q) * legendre_symbol(q, p) == sign


def test_supplementary_laws():
    for p in ODD_PRIMES:
        assert legendre_symbol(-1, p) == (1 if p % 4 == 1 else -1)
        assert legendre_symbol(2, p) == (1 if p % 8 in (1, 7) else -1)


def test_jacobi_symbol_is_product_of_legendre_symbols():
    for n in range(3, 400, 2):
        for a in range(-5, 40):
            expected = 1
            for p, e in prime_factorization(n).items():
                expected *= legendre_symbol(a, p) ** e
            assert jacobi_symbol(a, n) == expected


def test_jacobi_symbol_rejects_even_modulus():
    with pytest.raises(ValueError):
        jacobi_symbol(3, 10)


def test_sqrt_mod_squares_back_for_every_residue():
    for p in ODD_PRIMES:
        for a in range(p):
            if legendre_symbol(a, p) != -1:
                x = sqrt_mod(a, p)
                assert x * x % p == a
                assert 0 <= x <= p // 2


def test_sqrt_mod_large_prime_with_high_two_adicity():
    p = 998244353  # p - 1 = 2^23 * 7 * 17
    a = 123456789**2 % p
    assert sqrt_mod(a, p) in (123456789, p - 123456789)


def test_sqrt_mod_rejects_non_residue():
    with pytest.raises(ValueError):
        sqrt_mod(3, 7)

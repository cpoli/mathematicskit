"""Tests for primality testing and prime generation against known
primes/composites."""

import numpy as np
import pytest

from mathematicskit.number_theory.systems.primality import is_prime_miller_rabin, is_prime_trial_division, sieve_of_eratosthenes

KNOWN_PRIMES = [2, 3, 5, 7, 11, 97, 101, 7919, 2**31 - 1]
KNOWN_COMPOSITES = [1, 0, -5, 4, 9, 100, 91, 561, 2**31]  # 561 is a Carmichael number


@pytest.mark.parametrize("n", KNOWN_PRIMES)
def test_trial_division_identifies_known_primes(n):
    assert is_prime_trial_division(n)


@pytest.mark.parametrize("n", KNOWN_COMPOSITES)
def test_trial_division_identifies_known_composites(n):
    assert not is_prime_trial_division(n)


@pytest.mark.parametrize("n", KNOWN_PRIMES)
def test_miller_rabin_identifies_known_primes(n):
    assert is_prime_miller_rabin(n)


@pytest.mark.parametrize("n", KNOWN_COMPOSITES)
def test_miller_rabin_identifies_known_composites(n):
    assert not is_prime_miller_rabin(n)


def test_miller_rabin_agrees_with_trial_division_over_a_range():
    for n in range(2, 2000):
        assert is_prime_miller_rabin(n) == is_prime_trial_division(n)


def test_sieve_matches_trial_division():
    primes = sieve_of_eratosthenes(500)
    expected = [n for n in range(2, 501) if is_prime_trial_division(n)]
    np.testing.assert_array_equal(primes, expected)


def test_sieve_of_small_limit():
    np.testing.assert_array_equal(sieve_of_eratosthenes(1), [])
    np.testing.assert_array_equal(sieve_of_eratosthenes(2), [2])


def test_prime_counting_pi_100_is_25():
    assert sieve_of_eratosthenes(100).shape[0] == 25


# Mersenne primes past 2**63, where a numpy-drawn Miller-Rabin witness would
# overflow int64 rather than test the number.
LARGE_MERSENNE_PRIMES = [2**61 - 1, 2**89 - 1, 2**107 - 1, 2**127 - 1]


@pytest.mark.parametrize("n", LARGE_MERSENNE_PRIMES)
def test_miller_rabin_handles_arbitrary_precision_integers(n):
    """Witnesses must come from Python's arbitrary-precision RNG.

    ``numpy.random.Generator.integers`` is capped at int64, so drawing
    witnesses from it raised ``ValueError: high is out of bounds for int64``
    for every n >= 2**63 -- exactly the range where a probabilistic test is
    the only practical option."""
    assert is_prime_miller_rabin(n)


@pytest.mark.parametrize("n", [2**89 - 3, 2**107 - 3, (2**61 - 1) * (2**89 - 1)])
def test_miller_rabin_rejects_large_composites(n):
    assert not is_prime_miller_rabin(n)


@pytest.mark.parametrize("limit", [48, 49, 50, 120, 121, 122, 10**4])
def test_sieve_agrees_with_trial_division_across_square_boundaries(limit):
    """The sieve's inner bound must be an exact integer square root.

    ``int(np.sqrt(limit))`` can land one short on a perfect square once
    ``limit`` grows past float64's exact-integer range, leaving that prime's
    multiples unsieved; ``math.isqrt`` cannot."""
    assert list(sieve_of_eratosthenes(limit)) == [k for k in range(2, limit + 1) if is_prime_trial_division(k)]

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

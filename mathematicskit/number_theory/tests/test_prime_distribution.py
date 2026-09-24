"""Tests for the prime-counting function, logarithmic integral, primes in
arithmetic progressions, and Euler's product for zeta."""

import math

import numpy as np
import pytest
from scipy.special import zeta

from mathematicskit.number_theory.systems.primality import is_prime_trial_division
from mathematicskit.number_theory.systems.prime_distribution import logarithmic_integral, prime_counting, primes_in_progression
from mathematicskit.number_theory.systems.zeta import euler_product


def test_prime_counting_known_values():
    np.testing.assert_array_equal(prime_counting([0, 1, 2, 10, 100, 1000, 10**5, 10**6]), [0, 0, 1, 4, 25, 168, 9592, 78498])


def test_logarithmic_integral_known_values():
    assert logarithmic_integral(math.e) == pytest.approx(1.8951178163559368)
    assert logarithmic_integral(1.4513692348833810) == pytest.approx(0.0, abs=1e-12)  # Soldner's constant


def test_li_beats_x_over_log_x():
    x = 10**6
    assert abs(logarithmic_integral(x) - prime_counting(x)) < abs(x / math.log(x) - prime_counting(x))


def test_primes_in_progression_are_primes_in_the_class():
    primes = primes_in_progression(1, 10, 2000)
    assert all(p % 10 == 1 and is_prime_trial_division(int(p)) for p in primes)
    assert len(primes) == sum(1 for n in range(1, 2001, 10) if is_prime_trial_division(n))


def test_coprime_classes_partition_the_odd_primes():
    limit = 10**5
    total = sum(len(primes_in_progression(a, 12, limit)) for a in (1, 5, 7, 11))
    assert total == prime_counting(limit) - 2  # all primes except 2 and 3


def test_euler_product_converges_to_zeta():
    for s in (2.0, 3.0, 4.0):
        assert euler_product(s, 10**5) == pytest.approx(zeta(s), rel=1e-5)
    assert euler_product(2.0, 10**5) == pytest.approx(math.pi**2 / 6, rel=1e-5)


def test_euler_product_at_s_equals_1_grows_like_mertens():
    # Mertens' third theorem: prod_{p <= N} (1 - 1/p)^{-1} ~ e^gamma ln N.
    n = 10**6
    assert euler_product(1.0, n) == pytest.approx(math.exp(np.euler_gamma) * math.log(n), rel=1e-3)

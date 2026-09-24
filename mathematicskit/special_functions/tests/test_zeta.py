"""Tests for the Riemann zeta function and Euler's product against closed-form values."""

import math

import numpy as np
import pytest

from mathematicskit.special_functions.systems.zeta import _primes_up_to, euler_product, riemann_zeta


def test_even_values_basel_and_beyond():
    assert riemann_zeta(2.0) == pytest.approx(math.pi**2 / 6)
    assert riemann_zeta(4.0) == pytest.approx(math.pi**4 / 90)
    assert riemann_zeta(6.0) == pytest.approx(math.pi**6 / 945)


def test_matches_partial_sums_for_s_greater_than_one():
    n = np.arange(1, 200001, dtype=float)
    assert riemann_zeta(3.0) == pytest.approx(np.sum(n**-3.0), rel=1e-9)


def test_analytic_continuation_values():
    assert riemann_zeta(0.0) == pytest.approx(-0.5)
    assert riemann_zeta(-1.0) == pytest.approx(-1 / 12)
    for n in (1, 2, 3, 5):
        assert riemann_zeta(-2.0 * n) == pytest.approx(0.0, abs=1e-12)


def test_functional_equation():
    """zeta(s) = 2^s pi^{s-1} sin(pi s / 2) Gamma(1 - s) zeta(1 - s)."""
    for s in (-2.5, -0.3, 0.4, 3.2):
        rhs = 2**s * math.pi ** (s - 1) * math.sin(math.pi * s / 2) * math.gamma(1 - s) * riemann_zeta(1 - s)
        assert riemann_zeta(s) == pytest.approx(rhs, rel=1e-10)


def test_sieve():
    assert list(_primes_up_to(30)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_euler_product_converges_to_zeta():
    errors = [abs(euler_product(2.0, bound) - riemann_zeta(2.0)) for bound in (10, 100, 1000, 10000)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-4
    assert euler_product(3.0, 10000) == pytest.approx(riemann_zeta(3.0), rel=1e-8)

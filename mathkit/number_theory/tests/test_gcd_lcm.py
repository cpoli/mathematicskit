"""Tests for gcd/lcm against closed-form/known results."""

import numpy as np
import pytest

from mathkit.number_theory.utils.gcd_lcm import gcd, lcm


def test_gcd_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = int(rng.integers(1, 10000)), int(rng.integers(1, 10000))
        assert gcd(a, b) == np.gcd(a, b)


def test_gcd_known_values():
    assert gcd(48, 18) == 6
    assert gcd(17, 5) == 1
    assert gcd(0, 5) == 5


def test_lcm_known_values():
    assert lcm(4, 6) == 12
    assert lcm(21, 6) == 42
    assert lcm(0, 5) == 0


def test_gcd_times_lcm_equals_product():
    for a, b in [(4, 6), (12, 18), (7, 11)]:
        assert gcd(a, b) * lcm(a, b) == a * b


@pytest.mark.parametrize("a,b", [(-4, 6), (4, -6), (-4, -6)])
def test_gcd_ignores_sign(a, b):
    assert gcd(a, b) == gcd(abs(a), abs(b))

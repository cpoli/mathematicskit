"""Tests for the extended Euclidean algorithm, modular inverses, and
fast modular exponentiation against closed-form/known results."""

import numpy as np
import pytest

from mathematicskit.number_theory.systems.modular_arithmetic import extended_gcd, fast_mod_pow, mod_inverse


def test_extended_gcd_matches_bezout_identity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = int(rng.integers(1, 10000)), int(rng.integers(1, 10000))
        result = extended_gcd(a, b)
        assert a * result.x + b * result.y == result.gcd
        assert result.gcd == np.gcd(a, b)


def test_extended_gcd_handles_zero():
    result = extended_gcd(0, 5)
    assert result.gcd == 5
    result2 = extended_gcd(5, 0)
    assert result2.gcd == 5


def test_mod_inverse_matches_definition():
    for m in (7, 11, 26, 97):
        for a in range(1, m):
            if np.gcd(a, m) == 1:
                inv = mod_inverse(a, m)
                assert (a * inv) % m == 1


def test_mod_inverse_raises_when_not_coprime():
    with pytest.raises(ValueError):
        mod_inverse(4, 8)


def test_fast_mod_pow_matches_python_builtin():
    rng = np.random.default_rng(1)
    for _ in range(30):
        base, exp, mod = (int(rng.integers(1, 1000)) for _ in range(3))
        mod = max(mod, 1)
        assert fast_mod_pow(base, exp, mod) == pow(base, exp, mod)


def test_fast_mod_pow_zero_exponent_is_one_mod_m():
    assert fast_mod_pow(5, 0, 7) == 1


def test_fast_mod_pow_rejects_negative_exponent():
    with pytest.raises(ValueError):
        fast_mod_pow(2, -1, 5)

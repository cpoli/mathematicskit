"""Tests for Mathieu functions against the q = 0 limit and their differential equation."""

import numpy as np
import pytest

from mathematicskit.special_functions.systems.mathieu import mathieu_characteristic_a, mathieu_characteristic_b, mathieu_even, mathieu_odd


@pytest.mark.parametrize("m", [0, 1, 2, 3, 5])
def test_characteristic_values_at_q_zero_are_squares(m):
    assert mathieu_characteristic_a(m, 0.0) == pytest.approx(m**2)
    if m >= 1:
        assert mathieu_characteristic_b(m, 0.0) == pytest.approx(m**2)


def test_q_zero_reduces_to_trig():
    x = np.linspace(0, 2 * np.pi, 25)
    for m in (1, 2, 3):
        np.testing.assert_allclose(mathieu_even(m, 0.0, x), np.cos(m * x), atol=1e-12)
        np.testing.assert_allclose(mathieu_odd(m, 0.0, x), np.sin(m * x), atol=1e-12)


def test_small_q_expansion_of_a1():
    """a_1(q) = 1 + q - q^2/8 - q^3/64 + O(q^4)."""
    q = 0.05
    assert mathieu_characteristic_a(1, q) == pytest.approx(1 + q - q**2 / 8 - q**3 / 64, abs=1e-6)


def test_characteristic_values_are_ordered():
    """a_0 < b_1 < a_1 < b_2 < a_2 for q > 0."""
    q = 2.0
    values = [
        mathieu_characteristic_a(0, q),
        mathieu_characteristic_b(1, q),
        mathieu_characteristic_a(1, q),
        mathieu_characteristic_b(2, q),
        mathieu_characteristic_a(2, q),
    ]
    assert values == sorted(values)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_ce_satisfies_mathieu_equation(m):
    """y'' + (a - 2q cos 2x) y = 0, via finite differences."""
    q, h = 1.5, 1e-3
    a = mathieu_characteristic_a(m, q)
    for x in (0.3, 1.1, 2.4):
        y, yp, ym = (mathieu_even(m, q, x + d) for d in (0.0, h, -h))
        assert (yp - 2 * y + ym) / h**2 + (a - 2 * q * np.cos(2 * x)) * y == pytest.approx(0.0, abs=1e-4)


def test_ce_is_periodic_and_even():
    q = 3.0
    x = np.linspace(0.1, 3.0, 9)
    np.testing.assert_allclose(mathieu_even(2, q, x + np.pi), mathieu_even(2, q, x), atol=1e-8)
    np.testing.assert_allclose(mathieu_even(2, q, -x), mathieu_even(2, q, x), atol=1e-8)

"""Tests for the hypergeometric functions against classical closed forms."""

import math

import numpy as np
import pytest
from scipy import special

from mathematicskit.special_functions.systems.hypergeometric import confluent_hypergeometric_1f1, hypergeometric_2f1


def test_2f1_elementary_specializations():
    z = np.linspace(-0.9, 0.9, 19)
    np.testing.assert_allclose(z * hypergeometric_2f1(1.0, 1.0, 2.0, -z), np.log1p(z), atol=1e-14)
    np.testing.assert_allclose(hypergeometric_2f1(-2.5, 1.0, 1.0, -z), (1 + z) ** 2.5, rtol=1e-13)
    zz = np.linspace(0.0, 0.9, 10)
    np.testing.assert_allclose(zz * hypergeometric_2f1(0.5, 0.5, 1.5, zz**2), np.arcsin(zz), atol=1e-14)


@pytest.mark.parametrize(("a", "b", "c"), [(0.5, 0.25, 2.0), (1.0, 1.0, 3.0), (-1.5, 2.0, 1.2)])
def test_gauss_summation_theorem(a, b, c):
    expected = special.gamma(c) * special.gamma(c - a - b) / (special.gamma(c - a) * special.gamma(c - b))
    assert hypergeometric_2f1(a, b, c, 1.0) == pytest.approx(expected)


def test_2f1_terminates_for_negative_integer_a():
    """a = -n gives a polynomial; e.g. 2F1(-n, n; 1/2; (1-x)/2) = T_n(x)."""
    x = np.linspace(-1, 1, 11)
    n = 4
    np.testing.assert_allclose(hypergeometric_2f1(-n, n, 0.5, (1 - x) / 2), np.cos(n * np.arccos(x)), atol=1e-12)


def test_1f1_equal_parameters_is_exponential():
    z = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(confluent_hypergeometric_1f1(1.7, 1.7, z), np.exp(z), rtol=1e-13)


def test_kummer_transformation():
    a, b = 0.6, 2.3
    z = np.linspace(-4, 4, 17)
    np.testing.assert_allclose(confluent_hypergeometric_1f1(a, b, z), np.exp(z) * confluent_hypergeometric_1f1(b - a, b, -z), rtol=1e-12)


def test_1f1_gives_error_function():
    """erf(x) = (2x/sqrt(pi)) 1F1(1/2; 3/2; -x^2)."""
    x = 0.8
    assert 2 * x / math.sqrt(math.pi) * confluent_hypergeometric_1f1(0.5, 1.5, -(x**2)) == pytest.approx(math.erf(x))


def test_1f1_satisfies_kummer_equation():
    """z w'' + (b - z) w' - a w = 0, checked with finite differences."""
    a, b, z, h = 0.7, 1.9, 1.3, 1e-4
    w = confluent_hypergeometric_1f1(a, b, z)
    wp_ = confluent_hypergeometric_1f1(a, b, z + h)
    wm = confluent_hypergeometric_1f1(a, b, z - h)
    residual = z * (wp_ - 2 * w + wm) / h**2 + (b - z) * (wp_ - wm) / (2 * h) - a * w
    assert abs(residual) < 1e-5


def test_confluent_limit_of_2f1():
    """2F1(a, c; b; z/c) -> 1F1(a; b; z) as c -> inf, with O(1/c) error."""
    a, b, z = 0.5, 1.5, 0.7
    target = confluent_hypergeometric_1f1(a, b, z)
    errors = [abs(hypergeometric_2f1(a, c, b, z / c) - target) for c in (1e2, 1e3, 1e4)]
    assert errors[1] == pytest.approx(errors[0] / 10, rel=0.05)
    assert errors[2] == pytest.approx(errors[1] / 10, rel=0.05)

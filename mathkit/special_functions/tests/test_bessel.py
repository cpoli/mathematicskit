"""Tests for Bessel functions against closed-form/known results and
their defining differential equation."""

import numpy as np
import pytest

from mathkit.special_functions.systems.bessel import bessel_first_kind, bessel_second_kind


def test_j0_at_zero_is_one():
    assert bessel_first_kind(0.0, 0.0) == pytest.approx(1.0)


def test_jn_at_zero_is_zero_for_positive_order():
    for nu in (1.0, 2.0, 3.0):
        assert bessel_first_kind(nu, 0.0) == pytest.approx(0.0, abs=1e-10)


def test_bessel_satisfies_its_differential_equation():
    """x^2 y'' + x y' + (x^2 - nu^2) y = 0, checked via finite differences."""
    nu = 1.0
    x = 3.0
    h = 1e-4
    y = bessel_first_kind(nu, x)
    y_plus = bessel_first_kind(nu, x + h)
    y_minus = bessel_first_kind(nu, x - h)
    y_prime = (y_plus - y_minus) / (2 * h)
    y_double_prime = (y_plus - 2 * y + y_minus) / h**2
    residual = x**2 * y_double_prime + x * y_prime + (x**2 - nu**2) * y
    assert abs(residual) < 1e-3


def test_y0_diverges_toward_zero():
    assert bessel_second_kind(0.0, 0.01) < bessel_second_kind(0.0, 1.0)


def test_first_zero_of_j0_matches_known_value():
    """The first positive zero of J_0 is approximately 2.4048."""
    xs = np.linspace(2.0, 3.0, 10000)
    values = bessel_first_kind(0.0, xs)
    sign_changes = np.where(np.diff(np.sign(values)) != 0)[0]
    zero_crossing = xs[sign_changes[0]]
    assert zero_crossing == pytest.approx(2.4048, abs=1e-3)

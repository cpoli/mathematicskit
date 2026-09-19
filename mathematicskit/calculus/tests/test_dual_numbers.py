"""Tests for forward-mode automatic differentiation via dual numbers,
against closed-form derivatives."""

import math

import pytest

from mathematicskit.calculus.systems.dual_numbers import Dual, derivative


def test_polynomial_derivative_is_exact():
    f = lambda x: x * x * x - 2.0 * x + 1.0  # f' = 3x^2 - 2
    for x0 in (-2.0, 0.0, 1.5, 3.0):
        assert derivative(f, x0) == pytest.approx(3.0 * x0**2 - 2.0, abs=1e-10)


def test_trig_derivative_matches_closed_form():
    f = lambda x: x.sin()
    x0 = 0.7
    assert derivative(f, x0) == pytest.approx(math.cos(x0), abs=1e-12)


def test_chain_rule_composition():
    f = lambda x: (x * x).exp()  # f = exp(x^2), f' = 2x exp(x^2)
    x0 = 1.3
    expected = 2.0 * x0 * math.exp(x0**2)
    assert derivative(f, x0) == pytest.approx(expected, rel=1e-10)


def test_quotient_rule():
    f = lambda x: x / (x + 1.0)  # f' = 1/(x+1)^2
    x0 = 2.0
    assert derivative(f, x0) == pytest.approx(1.0 / (x0 + 1.0) ** 2, abs=1e-10)


def test_dual_arithmetic_matches_product_rule_directly():
    x = Dual(3.0, 1.0)
    y = x * x + 2.0 * x  # f = x^2 + 2x, f' = 2x + 2
    assert y.real == pytest.approx(15.0)
    assert y.dual == pytest.approx(8.0)


def test_no_finite_difference_truncation_error():
    """Unlike finite differences, dual numbers give the exact derivative
    to machine precision regardless of step size (there is no h)."""
    f = lambda x: x.log()
    x0 = 2.0
    assert derivative(f, x0) == pytest.approx(1.0 / x0, abs=1e-14)

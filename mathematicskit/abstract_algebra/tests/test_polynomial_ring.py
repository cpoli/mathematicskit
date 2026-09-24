"""Tests for polynomial ring arithmetic against closed-form/known results."""

import operator
from fractions import Fraction

import pytest

from mathematicskit.abstract_algebra.core.base import Polynomial
from mathematicskit.abstract_algebra.systems.polynomial_ring import poly_add, poly_divmod, poly_gcd, poly_mul, poly_sub


def test_add_matches_hand_computation():
    p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
    q = Polynomial([4, 5])  # 4 + 5x
    result = poly_add(p, q)
    assert result.coeffs == [Fraction(5), Fraction(7), Fraction(3)]


def test_sub_matches_hand_computation():
    p = Polynomial([5, 5])
    q = Polynomial([2, 3])
    result = poly_sub(p, q)
    assert result.coeffs == [Fraction(3), Fraction(2)]


def test_mul_matches_hand_computation():
    p = Polynomial([1, 1])  # 1 + x
    q = Polynomial([1, -1])  # 1 - x
    result = poly_mul(p, q)  # 1 - x^2
    assert result.coeffs == [Fraction(1), Fraction(0), Fraction(-1)]


def test_divmod_exact_division():
    p = Polynomial([-1, 0, 1])  # x^2 - 1
    q = Polynomial([-1, 1])  # x - 1
    quotient, remainder = poly_divmod(p, q)
    assert quotient.coeffs == [Fraction(1), Fraction(1)]  # x + 1
    assert remainder.degree == -1


def test_divmod_with_nonzero_remainder():
    p = Polynomial([1, 0, 1])  # x^2 + 1
    q = Polynomial([1, 1])  # x + 1
    quotient, remainder = poly_divmod(p, q)
    reconstructed = quotient * q + remainder
    assert reconstructed.coeffs == p.coeffs


def test_evaluate_matches_direct_computation():
    p = Polynomial([1, 2, 3])  # 1 + 2x + 3x^2
    assert p.evaluate(2) == 1 + 2 * 2 + 3 * 4


def test_gcd_of_x2_minus_1_and_x_minus_1():
    p = Polynomial([-1, 0, 1])  # x^2 - 1 = (x-1)(x+1)
    q = Polynomial([-1, 1])  # x - 1
    g = poly_gcd(p, q)
    assert g.coeffs == [Fraction(-1), Fraction(1)]  # x - 1 (monic)


def test_coprime_polynomials_have_gcd_one():
    p = Polynomial([1, 1])  # x + 1
    q = Polynomial([2, 1])  # x + 2
    g = poly_gcd(p, q)
    assert g.degree == 0


def test_gf2_polynomial_arithmetic():
    p = Polynomial([1, 1, 1], modulus=2)  # 1 + x + x^2
    q = Polynomial([1, 1], modulus=2)  # 1 + x
    result = poly_mul(p, q)
    # (1+x+x^2)(1+x) = 1 + x + x + x^2 + x^2 + x^3 = 1 + 0x + 0x^2 + x^3 mod 2
    assert result.coeffs == [1, 0, 0, 1]


def test_rejects_division_by_zero_polynomial():
    p = Polynomial([1, 1])
    zero = Polynomial([0])
    with pytest.raises(ZeroDivisionError):
        poly_divmod(p, zero)


@pytest.mark.parametrize("op", [operator.add, operator.sub, operator.mul, divmod])
def test_arithmetic_across_different_coefficient_fields_is_rejected(op):
    """GF(2) and GF(3) coefficients do not live in a common ring.

    Combining them used to succeed silently, reducing everything into the
    *left* operand's field and quietly producing a wrong answer."""
    with pytest.raises(ValueError, match="different fields"):
        op(Polynomial([1, 1], modulus=2), Polynomial([1, 1], modulus=3))
    with pytest.raises(ValueError, match="different fields"):
        op(Polynomial([1, 1]), Polynomial([1, 1], modulus=2))

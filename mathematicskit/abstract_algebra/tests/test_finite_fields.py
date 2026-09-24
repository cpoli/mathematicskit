"""Tests for finite field arithmetic against field-axiom checks and
known irreducibility results."""

import itertools

import pytest

from mathematicskit.abstract_algebra.core.base import Polynomial
from mathematicskit.abstract_algebra.systems.finite_fields import GF, find_irreducible_polynomial, is_irreducible


def test_gfp_inverse_satisfies_definition():
    field = GF(7)
    for a in range(1, 7):
        inv = field.inverse(a)
        assert field.multiply(a, inv) == 1


def test_gfp_rejects_zero_inverse():
    field = GF(5)
    with pytest.raises(ZeroDivisionError):
        field.inverse(0)


def test_gfp_addition_and_multiplication_wrap_around():
    field = GF(5)
    assert field.add(3, 4) == 2
    assert field.multiply(3, 4) == 2


@pytest.mark.parametrize("p,n", [(2, 2), (2, 3), (3, 2), (5, 2)])
def test_gf_extension_every_nonzero_element_has_inverse(p, n):
    field = GF(p, n)
    zero = Polynomial([0], modulus=p)
    one = Polynomial([1], modulus=p)
    for e in field.elements():
        if e == zero:
            continue
        inv = field.inverse(e)
        assert field.multiply(e, inv) == one


def test_gf_extension_order_is_p_to_the_n():
    field = GF(2, 3)
    assert field.order == 8
    assert len(field.elements()) == 8


def test_is_irreducible_matches_brute_force_factorization_over_gf2():
    """A degree-3 polynomial over GF(2) is irreducible iff it has no root in GF(2)."""
    for coeffs in itertools.product([0, 1], repeat=3):
        poly = Polynomial(list(coeffs) + [1], modulus=2)
        has_root = any(poly.evaluate(x) == 0 for x in (0, 1))
        assert is_irreducible(poly) == (not has_root)


def test_find_irreducible_polynomial_is_monic_and_correct_degree():
    for p, n in [(2, 2), (2, 4), (3, 3)]:
        poly = find_irreducible_polynomial(p, n)
        assert poly.degree == n
        assert poly.coeffs[-1] == 1
        assert is_irreducible(poly)


def test_gf_multiplication_stays_within_extension_degree():
    field = GF(2, 3)
    elements = field.elements()
    for a in elements:
        for b in elements:
            product = field.multiply(a, b)
            assert product.degree < 3


@pytest.mark.parametrize("coeffs", [[0], [1], [2]])
def test_units_and_the_zero_polynomial_are_not_irreducible(coeffs):
    """Irreducibility asks for a non-unit with no non-unit factorisation.

    Degree-0 polynomials are units and the zero polynomial is neither, so
    neither qualifies -- but a bare ``degree <= 1`` shortcut reported both
    as irreducible."""
    assert not is_irreducible(Polynomial(coeffs, modulus=3))


@pytest.mark.parametrize("p,n", [(2, 2), (2, 3), (2, 4), (3, 2), (5, 2)])
def test_degree_one_polynomials_remain_irreducible(p, n):
    """The tightened guard must not sweep up genuine degree-1 irreducibles."""
    assert is_irreducible(Polynomial([1, 1], modulus=p))
    assert find_irreducible_polynomial(p, n).degree == n

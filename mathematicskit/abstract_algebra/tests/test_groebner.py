"""Tests for multivariate polynomials, division, and Buchberger's algorithm."""

from fractions import Fraction

import pytest

from mathematicskit.abstract_algebra.core.base import MultivariatePolynomial
from mathematicskit.abstract_algebra.systems.groebner import groebner_basis, in_ideal, multivariate_divide, s_polynomial


@pytest.fixture
def xyz():
    return MultivariatePolynomial.variables(3)


def test_polynomial_arithmetic_and_evaluation():
    x, y = MultivariatePolynomial.variables(2)
    f = (x + 2 * y) * (x - y) - 3
    assert f == x**2 + x * y - 2 * y**2 - 3
    assert f.evaluate([2, 1]) == Fraction(1)
    assert (f - f).is_zero() and f.total_degree == 2
    assert 1 - x == -(x - 1)
    with pytest.raises(ValueError):
        x + MultivariatePolynomial.variables(3)[0]
    with pytest.raises(ValueError):
        (x - x).leading_term()


def test_monomial_orders_differ():
    x, y, z = MultivariatePolynomial.variables(3)
    f = x * z**2 + y**3
    assert f.leading_term("lex")[0] == (1, 0, 2)
    assert f.leading_term("grlex")[0] == (1, 0, 2)  # tie on degree 3, broken by lex
    assert f.leading_term("grevlex")[0] == (0, 3, 0)  # grevlex: smaller power of the last variable wins


def test_division_identity(xyz):
    x, y, z = xyz
    f = x**3 * y + 2 * x * y * z - z**2 + 5
    divisors = [x * y - z, x**2 + y - 1]
    q, r = multivariate_divide(f, divisors, "grlex")
    assert sum((qi * gi for qi, gi in zip(q, divisors, strict=True)), r) == f
    leads = [g.leading_term("grlex")[0] for g in divisors]
    assert all(not all(a <= b for a, b in zip(lead, e, strict=True)) for e in r.terms for lead in leads)


def test_division_remainder_depends_on_divisor_order_but_not_for_groebner_basis():
    x, y = MultivariatePolynomial.variables(2)
    f = x**2 * y + x * y**2 + y**2
    g1, g2 = x * y - 1, y**2 - 1
    r12 = multivariate_divide(f, [g1, g2])[1]
    r21 = multivariate_divide(f, [g2, g1])[1]
    assert r12 != r21
    basis = groebner_basis([g1, g2]).basis
    assert multivariate_divide(f, basis)[1] == multivariate_divide(f, basis[::-1])[1]


def test_s_polynomials_of_groebner_basis_reduce_to_zero(xyz):
    x, y, z = xyz
    for order in ("lex", "grlex", "grevlex"):
        g = groebner_basis([x**2 + y**2 + z**2 - 1, x - y, y * z - 1], order).basis
        for i in range(len(g)):
            for j in range(i):
                assert multivariate_divide(s_polynomial(g[i], g[j], order), g, order)[1].is_zero()


def test_lex_basis_is_triangular_and_solves_system():
    x, y = MultivariatePolynomial.variables(2)
    result = groebner_basis([x**2 + y**2 - 4, x * y - 1], "lex")
    last = result.basis[-1]
    assert all(e[0] == 0 for e in last.terms)  # eliminated x
    assert last == y**4 - 4 * y**2 + 1
    assert result.s_pairs >= 1


def test_twisted_cubic_bases(xyz):
    # The twisted cubic (t, t^2, t^3) is cut out by <x^2 - y, x^3 - z>.
    x, y, z = xyz
    assert set(groebner_basis([x**2 - y, x**3 - z], "lex").basis) == {x**2 - y, x * y - z, x * z - y**2, y**3 - z**2}
    basis = groebner_basis([x**2 - y, x**3 - z], "grlex").basis
    expected = {x**2 - y, x * y - z, x * z - y**2, y**3 - z**2}
    assert set(basis) == expected


def test_ideal_membership(xyz):
    x, y, z = xyz
    g = groebner_basis([x * z - y**2, x**3 - z**2], "grlex")
    assert in_ideal((x * z - y**2) * (x + 3) + (x**3 - z**2) * y * z, g)
    assert not in_ideal(x * y - 5 * z**2 + x, g)


def test_unit_ideal_and_finite_field():
    x, y = MultivariatePolynomial.variables(2)
    assert groebner_basis([x * y - 1, x]).basis == [x * 0 + 1]
    a, b = MultivariatePolynomial.variables(2, modulus=5)
    basis = groebner_basis([a**2 + b**2 - 1, a - 2 * b]).basis
    assert all(p.modulus == 5 for p in basis)
    # Over GF(5): a = 2b, 5b^2 = 0 = 1 is impossible, so the ideal is the unit ideal.
    assert basis == [a * 0 + 1]


def test_rejects_empty_input():
    x, _ = MultivariatePolynomial.variables(2)
    with pytest.raises(ValueError):
        groebner_basis([x - x])


def test_agrees_with_sympy(xyz):
    sympy = pytest.importorskip("sympy")
    x, y, z = xyz
    polys = [x**2 + y * z - 2, x * z + y**2 - 3, x * y + z**2 - 5]
    sx, sy, sz = sympy.symbols("x0 x1 x2")
    sp = [sx**2 + sy * sz - 2, sx * sz + sy**2 - 3, sx * sy + sz**2 - 5]
    for order in ("lex", "grevlex"):
        ours = groebner_basis(polys, order).basis
        theirs = sympy.groebner(sp, sx, sy, sz, order=order).exprs
        gens = (sx, sy, sz)
        mine = {sympy.Poly(sympy.sympify(repr(p).replace("^", "**")), *gens).monic() for p in ours}
        assert mine == {sympy.Poly(e, *gens).monic() for e in theirs}

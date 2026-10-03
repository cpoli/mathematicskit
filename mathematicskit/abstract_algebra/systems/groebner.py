r"""Gröbner bases by Buchberger's algorithm, and multivariate division.

Dividing a polynomial by several others, term by term, generalizes long
division, but the remainder depends on the order of the divisors and
can be nonzero even for members of the ideal they generate. Buchberger
(1965) called a generating set a Gröbner basis when every member of the
ideal has remainder zero, and showed how to complete any generating set
to one: add the remainder of every S-polynomial

.. math::

   S(f, g) = \frac{L}{\operatorname{LT}(f)}\,f - \frac{L}{\operatorname{LT}(g)}\,g,
   \qquad L = \operatorname{lcm}(\operatorname{LM}(f), \operatorname{LM}(g)),

which cancels the two leading terms, until all of them reduce to zero.
A lexicographic Gröbner basis is triangular, like the echelon form of a
linear system, which makes it an elimination method for polynomial
systems. Hand-rolled on
:class:`~mathematicskit.abstract_algebra.core.base.MultivariatePolynomial`,
in exact arithmetic over :math:`\mathbb Q` or :math:`\mathrm{GF}(p)`.
See B. Buchberger, *Ein Algorithmus zum Auffinden der Basiselemente des
Restklassenringes nach einem nulldimensionalen Polynomideal* (PhD
thesis, Universität Innsbruck, 1965), and Cox, Little & O'Shea, *Ideals,
Varieties, and Algorithms*, 4th ed. (2015), Ch. 2-3.
"""

from __future__ import annotations

from collections.abc import Sequence

from mathematicskit.abstract_algebra.core.base import _MONOMIAL_ORDERS, GroebnerResult, MultivariatePolynomial

__all__ = ["multivariate_divide", "s_polynomial", "groebner_basis", "in_ideal"]


def _divides(a: tuple, b: tuple) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True))


def _monomial(f: MultivariatePolynomial, exponent: tuple, coefficient) -> MultivariatePolynomial:
    return MultivariatePolynomial({exponent: coefficient}, f.n_vars, f.modulus)


def multivariate_divide(f: MultivariatePolynomial, divisors: Sequence[MultivariatePolynomial], order: str = "lex") -> tuple:
    r"""Divide ``f`` by ``divisors``: :math:`f = \sum_i q_i g_i + r`.

    The standard algorithm (Cox, Little & O'Shea, Sec. 2.3): repeatedly
    take the leading term of what is left; if some :math:`\operatorname{LT}(g_i)`
    divides it, cancel it with a multiple of :math:`g_i` (first such
    :math:`i`), otherwise move it to the remainder. No term of :math:`r`
    is divisible by any :math:`\operatorname{LT}(g_i)`.

    Parameters
    ----------
    f : MultivariatePolynomial
    divisors : sequence of MultivariatePolynomial
    order : {"lex", "grlex", "grevlex"}

    Returns
    -------
    tuple of (list of MultivariatePolynomial, MultivariatePolynomial)
        The quotients :math:`q_i` and the remainder :math:`r`.

    Examples
    --------
    >>> x, y = MultivariatePolynomial.variables(2)
    >>> quotients, r = multivariate_divide(x**2 * y + x * y**2 + y**2, [x * y - 1, y**2 - 1])
    >>> quotients, r
    ([x0 + x1, 1], x0 + x1 + 1)
    """
    zero = f * 0
    quotients = [zero for _ in divisors]
    leads = [g.leading_term(order) for g in divisors]
    remainder = zero
    p = f
    while not p.is_zero():
        e, c = p.leading_term(order)
        for i, (eg, cg) in enumerate(leads):
            if _divides(eg, e):
                factor = _monomial(f, tuple(a - b for a, b in zip(e, eg, strict=True)), c * divisors[i]._inverse(cg))
                quotients[i] = quotients[i] + factor
                p = p - factor * divisors[i]
                break
        else:
            lead = _monomial(f, e, c)
            remainder = remainder + lead
            p = p - lead
    return quotients, remainder


def s_polynomial(f: MultivariatePolynomial, g: MultivariatePolynomial, order: str = "lex") -> MultivariatePolynomial:
    r"""The S-polynomial :math:`S(f, g)`, in which the leading terms of ``f`` and ``g`` cancel.

    Parameters
    ----------
    f, g : MultivariatePolynomial
    order : {"lex", "grlex", "grevlex"}

    Returns
    -------
    MultivariatePolynomial

    Examples
    --------
    >>> x, y = MultivariatePolynomial.variables(2)
    >>> s_polynomial(x**3 * y**2 - x**2 * y**3 + x, 3 * x**4 * y + y**2, "grlex")
    -x0^3*x1^3 - 1/3*x1^3 + x0^2
    """
    (ef, cf), (eg, cg) = f.leading_term(order), g.leading_term(order)
    lcm = tuple(max(a, b) for a, b in zip(ef, eg, strict=True))
    left = _monomial(f, tuple(a - b for a, b in zip(lcm, ef, strict=True)), f._inverse(cf))
    right = _monomial(g, tuple(a - b for a, b in zip(lcm, eg, strict=True)), g._inverse(cg))
    return left * f - right * g


def _reduce_basis(basis: list, order: str) -> list:
    monic = [g.monic(order) for g in basis]
    minimal: list = []
    for i, g in enumerate(monic):  # drop elements whose leading monomial another one's divides
        eg = g.leading_term(order)[0]
        if not any(_divides(h.leading_term(order)[0], eg) and (h.leading_term(order)[0] != eg or j < i) for j, h in enumerate(monic) if j != i):
            minimal.append(g)
    reduced = []
    for i, g in enumerate(minimal):
        _, r = multivariate_divide(g, minimal[:i] + minimal[i + 1 :], order)
        reduced.append(r.monic(order))
    return sorted(reduced, key=lambda h: _order_key(h, order), reverse=True)


def _order_key(f: MultivariatePolynomial, order: str):
    return _MONOMIAL_ORDERS[order](f.leading_term(order)[0])


def groebner_basis(polynomials: Sequence[MultivariatePolynomial], order: str = "lex") -> GroebnerResult:
    r"""The reduced Gröbner basis of the ideal generated by ``polynomials``.

    Buchberger's algorithm: keep a queue of pairs; reduce each pair's
    S-polynomial by the current basis, and if the remainder is nonzero,
    add it and queue its pairs with every existing element. Pairs whose
    leading monomials are coprime are skipped, since their S-polynomial
    always reduces to zero (Buchberger's first criterion). The result is
    then made reduced, hence unique.

    Parameters
    ----------
    polynomials : sequence of MultivariatePolynomial
    order : {"lex", "grlex", "grevlex"}

    Returns
    -------
    GroebnerResult

    Examples
    --------
    >>> x, y = MultivariatePolynomial.variables(2)
    >>> # A circle meets a parabola: lex order eliminates x, leaving a polynomial in y alone.
    >>> groebner_basis([x**2 + y**2 - 1, y - x**2], "lex").basis
    [x0^2 - x1, x1^2 + x1 - 1]
    """
    basis = [p for p in polynomials if not p.is_zero()]
    if not basis:
        raise ValueError("need at least one nonzero polynomial")
    pairs = [(i, j) for i in range(len(basis)) for j in range(i)]
    s_pairs = zero_reductions = 0
    while pairs:
        i, j = pairs.pop(0)
        ei, ej = basis[i].leading_term(order)[0], basis[j].leading_term(order)[0]
        if all(a == 0 or b == 0 for a, b in zip(ei, ej, strict=True)):
            continue
        s_pairs += 1
        _, r = multivariate_divide(s_polynomial(basis[i], basis[j], order), basis, order)
        if r.is_zero():
            zero_reductions += 1
            continue
        basis.append(r)
        pairs.extend((len(basis) - 1, k) for k in range(len(basis) - 1))
    return GroebnerResult(basis=_reduce_basis(basis, order), order=order, s_pairs=s_pairs, zero_reductions=zero_reductions)


def in_ideal(f: MultivariatePolynomial, groebner: GroebnerResult) -> bool:
    r"""Whether ``f`` lies in the ideal: its remainder on division by a Gröbner basis is zero.

    Parameters
    ----------
    f : MultivariatePolynomial
    groebner : GroebnerResult

    Returns
    -------
    bool

    Examples
    --------
    >>> x, y = MultivariatePolynomial.variables(2)
    >>> g = groebner_basis([x * y - 1, y**2 - 1])
    >>> in_ideal(x - y, g), in_ideal(x + y, g)
    (True, False)
    """
    return multivariate_divide(f, groebner.basis, groebner.order)[1].is_zero()

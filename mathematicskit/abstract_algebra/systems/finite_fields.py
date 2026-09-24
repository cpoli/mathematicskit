r"""Finite field arithmetic: :math:`\mathrm{GF}(p)` and
:math:`\mathrm{GF}(p^n)` via irreducible polynomials over :math:`\mathrm{GF}(p)`.

No numpy/scipy equivalent. See Dummit & Foote, *Abstract Algebra*, 3rd
ed., Sec. 13.5 (finite fields) and Lidl & Niederreiter, *Introduction to
Finite Fields and Their Applications*, Ch. 2-3.
"""

from __future__ import annotations

import itertools
from typing import Optional

from mathematicskit.abstract_algebra.core.base import Polynomial

__all__ = ["GF", "is_irreducible", "find_irreducible_polynomial"]


def is_irreducible(poly: Polynomial) -> bool:
    r"""Whether `poly` is irreducible over :math:`\mathrm{GF}(p)`: has no polynomial factor of degree :math:`1 \leq d \leq \deg(\text{poly})/2`.

    Checked by trial division against every monic polynomial of degree
    ``1`` through ``deg(poly) // 2`` (degree-1 divisors are equivalent
    to checking for a root in :math:`\mathrm{GF}(p)`). Feasible for the
    small ``p``, small degree cases this domain's examples use. See
    Lidl & Niederreiter, *Introduction to Finite Fields*, Sec. 3.1.

    Parameters
    ----------
    poly : Polynomial
        With ``modulus`` set (coefficients in :math:`\mathrm{GF}(p)`).

    Returns
    -------
    bool

    Examples
    --------
    >>> is_irreducible(Polynomial([1, 1, 1], modulus=2))  # x^2+x+1 over GF(2)
    True
    >>> is_irreducible(Polynomial([0, 1, 1], modulus=2))  # x^2+x = x(x+1): reducible
    False
    >>> is_irreducible(Polynomial([1], modulus=2))  # the constant 1 is a unit, not irreducible
    False
    >>> is_irreducible(Polynomial([0], modulus=2))  # nor is the zero polynomial
    False
    """
    p = poly.modulus
    n = poly.degree
    if n < 1:
        # Degree 0 is a nonzero constant (a unit) and degree -1 is the zero
        # polynomial; by definition neither is irreducible, since
        # irreducibility asks for a non-unit with no non-unit factorization.
        return False
    if n == 1:
        return True
    for d in range(1, n // 2 + 1):
        for candidate_coeffs in _monic_polynomials_of_degree(p, d):
            candidate = Polynomial(candidate_coeffs, modulus=p)
            _, remainder = divmod(poly, candidate)
            if remainder.degree == -1:
                return False
    return True


def _monic_polynomials_of_degree(p: int, d: int):
    """Every monic polynomial of degree `d` over GF(p), lowest coefficient first."""
    if d == 0:
        yield [1]
        return
    for lower_coeffs in itertools.product(range(p), repeat=d):
        yield list(lower_coeffs) + [1]


def find_irreducible_polynomial(p: int, n: int) -> Polynomial:
    r"""Find a monic irreducible polynomial of degree ``n`` over :math:`\mathrm{GF}(p)`, by brute-force search.

    An irreducible polynomial of every degree exists over every finite
    field (Dummit & Foote, *Abstract Algebra*, 3rd ed., Corollary 13.24
    -- via counting arguments), so this search always terminates. See
    Lidl & Niederreiter, *Introduction to Finite Fields*, Sec. 3.1-3.3.

    Parameters
    ----------
    p : int
        Prime.
    n : int
        Degree, ``n >= 1``.

    Returns
    -------
    Polynomial

    Examples
    --------
    >>> poly = find_irreducible_polynomial(2, 3)
    >>> poly.degree
    3
    >>> is_irreducible(poly)
    True
    """
    if n == 1:
        return Polynomial([0, 1], modulus=p)
    for coeffs in _monic_polynomials_of_degree(p, n):
        candidate = Polynomial(coeffs, modulus=p)
        if is_irreducible(candidate):
            return candidate
    raise RuntimeError(f"no irreducible polynomial of degree {n} found over GF({p})")


class GF:
    r"""The finite field :math:`\mathrm{GF}(p^n)`.

    ``n=1`` (the default) is the prime field :math:`\mathrm{GF}(p) =
    \mathbb{Z}/p\mathbb{Z}`, with elements plain integers ``0, ..., p-1``.
    ``n>1`` represents elements as polynomials of degree ``< n`` over
    :math:`\mathrm{GF}(p)`, with multiplication reduced modulo a fixed
    irreducible polynomial (found via :func:`find_irreducible_polynomial`
    if not supplied) -- :math:`\mathrm{GF}(p)[x]/(f(x))` is a field
    exactly when ``f`` is irreducible. See Dummit & Foote, *Abstract
    Algebra*, 3rd ed., Sec. 13.5.

    Parameters
    ----------
    p : int
        Prime.
    n : int
        Extension degree, ``n >= 1``.
    irreducible : Polynomial, optional
        A degree-``n`` irreducible polynomial over :math:`\mathrm{GF}(p)`;
        found automatically if omitted.

    Examples
    --------
    >>> field = GF(2, 3)  # GF(8)
    >>> field.order
    8
    >>> a = Polynomial([1, 1, 0], modulus=2)  # 1 + x
    >>> b = Polynomial([0, 1, 1], modulus=2)  # x + x^2
    >>> field.multiply(a, b).degree <= 2  # result stays reduced mod the irreducible polynomial
    True
    """

    def __init__(self, p: int, n: int = 1, irreducible: Optional[Polynomial] = None):
        self.p = int(p)
        self.n = int(n)
        self.irreducible = irreducible if irreducible is not None else (find_irreducible_polynomial(p, n) if n > 1 else None)

    @property
    def order(self) -> int:
        """int: :math:`|\\mathrm{GF}(p^n)| = p^n`."""
        return self.p**self.n

    def elements(self) -> list:
        """Every field element.

        Returns
        -------
        list of int (if ``n == 1``) or list of Polynomial (if ``n > 1``)
        """
        if self.n == 1:
            return list(range(self.p))
        return [Polynomial(list(coeffs), modulus=self.p) for coeffs in itertools.product(range(self.p), repeat=self.n)]

    def add(self, a, b):
        """Field addition."""
        if self.n == 1:
            return (a + b) % self.p
        return a + b

    def multiply(self, a, b):
        """Field multiplication, reduced modulo the extension's irreducible polynomial when ``n > 1``."""
        if self.n == 1:
            return (a * b) % self.p
        product = a * b
        _, remainder = divmod(product, self.irreducible)
        return remainder

    def inverse(self, a):
        r"""Multiplicative inverse of a nonzero element.

        ``n == 1``: via Fermat's little theorem, :math:`a^{-1} \equiv
        a^{p-2} \pmod p`. ``n > 1``: via the polynomial extended
        Euclidean algorithm (finding :math:`u, v` with :math:`ua +
        v \cdot \text{irreducible} = \gcd = 1`).
        """
        if self.n == 1:
            if a % self.p == 0:
                raise ZeroDivisionError("0 has no multiplicative inverse")
            return pow(a, self.p - 2, self.p)
        if a.degree == -1:
            raise ZeroDivisionError("0 has no multiplicative inverse")
        return _poly_extended_gcd_inverse(a, self.irreducible, self.p)


def _poly_extended_gcd_inverse(a: Polynomial, modulus_poly: Polynomial, p: int) -> Polynomial:
    """Find a's inverse mod modulus_poly via the polynomial extended Euclidean algorithm."""
    zero = Polynomial([0], modulus=p)
    one = Polynomial([1], modulus=p)
    old_r, r = a, modulus_poly
    old_s, s = one, zero
    while r.degree != -1:
        q, remainder = divmod(old_r, r)
        old_r, r = r, remainder
        old_s, s = s, old_s - q * s
    # old_r is now gcd(a, modulus_poly), which must be a nonzero constant
    # since modulus_poly is irreducible; normalize to exactly 1.
    lead = old_r.coeffs[old_r.degree]
    lead_inv = pow(int(lead), p - 2, p)
    return Polynomial([c * lead_inv for c in old_s.coeffs], modulus=p)

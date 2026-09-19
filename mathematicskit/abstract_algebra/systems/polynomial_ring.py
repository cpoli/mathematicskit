r"""Polynomial ring arithmetic over :math:`\mathbb{Z}`, :math:`\mathbb{Q}`,
and finite fields.

Thin, documented entry points around
:class:`mathematicskit.abstract_algebra.core.base.Polynomial`'s arithmetic --
addition, multiplication, and division with remainder, hand-rolled from
the textbook long-division algorithm (no numpy/scipy equivalent). See
Dummit & Foote, *Abstract Algebra*, 3rd ed., Ch. 9.
"""

from __future__ import annotations

from typing import Optional

from mathematicskit.abstract_algebra.core.base import Polynomial

__all__ = ["poly_add", "poly_sub", "poly_mul", "poly_divmod", "poly_gcd"]


def poly_add(p: Polynomial, q: Polynomial) -> Polynomial:
    """Polynomial addition.

    Parameters
    ----------
    p, q : Polynomial

    Returns
    -------
    Polynomial

    Examples
    --------
    >>> poly_add(Polynomial([1, 2]), Polynomial([3, 4, 5])).coeffs
    [Fraction(4, 1), Fraction(6, 1), Fraction(5, 1)]
    """
    return p + q


def poly_sub(p: Polynomial, q: Polynomial) -> Polynomial:
    """Polynomial subtraction.

    Parameters
    ----------
    p, q : Polynomial

    Returns
    -------
    Polynomial
    """
    return p - q


def poly_mul(p: Polynomial, q: Polynomial) -> Polynomial:
    r"""Polynomial multiplication: convolution of coefficients.

    Parameters
    ----------
    p, q : Polynomial

    Returns
    -------
    Polynomial

    Examples
    --------
    >>> poly_mul(Polynomial([1, 1]), Polynomial([1, -1])).coeffs  # (1+x)(1-x) = 1 - x^2
    [Fraction(1, 1), Fraction(0, 1), Fraction(-1, 1)]
    """
    return p * q


def poly_divmod(p: Polynomial, q: Polynomial):
    r"""Polynomial division with remainder: ``p = quotient * q + remainder``, ``deg(remainder) < deg(q)``.

    Standard polynomial long division, requiring `q`'s leading
    coefficient to be invertible (automatic over :math:`\mathbb{Q}` or
    :math:`\mathrm{GF}(p)`, both fields). See Dummit & Foote, *Abstract
    Algebra*, 3rd ed., Sec. 9.2, "Division Algorithm".

    Parameters
    ----------
    p, q : Polynomial

    Returns
    -------
    (Polynomial, Polynomial)
        ``(quotient, remainder)``.

    Examples
    --------
    >>> quotient, remainder = poly_divmod(Polynomial([-1, 0, 1]), Polynomial([-1, 1]))  # (x^2-1)/(x-1)
    >>> quotient.coeffs, remainder.coeffs  # x + 1, remainder 0
    ([Fraction(1, 1), Fraction(1, 1)], [Fraction(0, 1)])
    """
    return divmod(p, q)


def poly_gcd(p: Polynomial, q: Polynomial) -> Polynomial:
    r"""Greatest common divisor of two polynomials, via the Euclidean algorithm.

    :math:`\gcd(p, q) = \gcd(q, p \bmod q)`, terminating when the
    remainder is the zero polynomial -- exactly the integer Euclidean
    algorithm with polynomial division in place of integer division. See
    Dummit & Foote, *Abstract Algebra*, 3rd ed., Sec. 9.2.

    Parameters
    ----------
    p, q : Polynomial
        Must share the same coefficient field (``modulus``).

    Returns
    -------
    Polynomial

    Examples
    --------
    >>> # gcd(x^2 - 1, x - 1) = x - 1 (up to a scalar).
    >>> g = poly_gcd(Polynomial([-1, 0, 1]), Polynomial([-1, 1]))
    >>> g.degree
    1
    """
    modulus: Optional[int] = p.modulus
    a, b = p, q
    while b.degree != -1:
        _, remainder = divmod(a, b)
        a, b = b, remainder
    if a.degree == -1:
        return a
    # Normalize to monic (leading coefficient 1) so the result is unique.
    lead = a.coeffs[a.degree]
    lead_inv = (1 / lead) if modulus is None else pow(int(lead), modulus - 2, modulus)
    return Polynomial([c * lead_inv if modulus is None else (c * lead_inv) % modulus for c in a.coeffs], modulus=modulus)

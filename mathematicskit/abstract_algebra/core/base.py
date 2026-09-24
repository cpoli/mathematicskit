r"""Abstract base class for finite groups, and a coefficient-generic
polynomial class, shared across mathematicskit.abstract_algebra's systems/
modules.

Every algorithm in this domain is hand-rolled from its textbook
definition -- group/ring/field theory over finite structures has no
``numpy``/``scipy`` equivalent at all. See Dummit & Foote, *Abstract
Algebra*, 3rd ed., throughout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

__all__ = ["FiniteGroup", "GroupPropertiesResult", "SylowResult", "CompositionSeriesResult", "HomomorphismResult", "Polynomial"]


class FiniteGroup(ABC):
    r"""Common interface for a finite group :math:`(G, \cdot)`.

    Concrete subclasses (:class:`~mathematicskit.abstract_algebra.systems.groups.CyclicGroup`,
    :class:`~mathematicskit.abstract_algebra.systems.groups.PermutationGroup`)
    provide the element set and the group operation; this base class
    derives the Cayley table, abelian check, and element orders purely
    from those. See Dummit & Foote, *Abstract Algebra*, 3rd ed., Ch. 1.

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> g = CyclicGroup(4)
    >>> g.order
    4
    >>> g.is_abelian()
    True
    """

    @property
    @abstractmethod
    def elements(self) -> list:
        """list: Every group element, in a fixed (implementation-defined) order."""

    @abstractmethod
    def identity(self):
        """The group's identity element."""

    @abstractmethod
    def operate(self, a, b):
        """The group operation, :math:`a \\cdot b`."""

    @abstractmethod
    def inverse(self, a):
        """The inverse of ``a``."""

    @property
    def order(self) -> int:
        """int: :math:`|G|`, the number of elements."""
        return len(self.elements)

    def cayley_table(self) -> np.ndarray:
        r"""The Cayley (multiplication) table, as element indices.

        Row ``i``, column ``j`` holds the index (into :attr:`elements`)
        of ``elements[i] . elements[j]``. See Dummit & Foote, *Abstract
        Algebra*, 3rd ed., Sec. 1.7.

        Returns
        -------
        ndarray, shape (n, n), int
        """
        els = self.elements
        index = {e: i for i, e in enumerate(els)}
        n = len(els)
        table = np.zeros((n, n), dtype=np.int64)
        for i, a in enumerate(els):
            for j, b in enumerate(els):
                table[i, j] = index[self.operate(a, b)]
        return table

    def is_abelian(self) -> bool:
        """Whether the group operation is commutative for every pair of elements.

        Returns
        -------
        bool
        """
        els = self.elements
        return all(self.operate(a, b) == self.operate(b, a) for a in els for b in els)

    def element_order(self, a) -> int:
        r"""The order of element ``a``: the smallest :math:`k>0` with :math:`a^k = e`.

        Parameters
        ----------
        a

        Returns
        -------
        int
        """
        e = self.identity()
        x = a
        k = 1
        while x != e:
            x = self.operate(x, a)
            k += 1
            if k > self.order:
                raise RuntimeError("element does not appear to have finite order within |G| steps")
        return k


@dataclass
class GroupPropertiesResult:
    """Container for a finite group's summary properties."""

    order: int
    is_abelian: bool
    identity: object
    element_orders: dict = field(default_factory=dict)
    """dict: ``{element: order}`` for every element."""


@dataclass
class SylowResult:
    """Container for the Sylow :math:`p`-subgroups of a finite group."""

    p: int
    sylow_order: int
    """int: :math:`p^k`, the largest power of ``p`` dividing :math:`|G|`."""
    subgroups: list
    """list of list: Every Sylow :math:`p`-subgroup, each as a list of elements."""

    @property
    def count(self) -> int:
        """int: :math:`n_p`, the number of Sylow :math:`p`-subgroups."""
        return len(self.subgroups)


@dataclass
class CompositionSeriesResult:
    """Container for a composition series :math:`G = G_0 > G_1 > \\dots > G_m = \\{e\\}`."""

    series: list
    """list of list: The subgroups :math:`G_0, \\dots, G_m`, largest first."""
    factor_orders: list
    """list of int: The orders :math:`|G_i / G_{i+1}|` of the composition factors."""


@dataclass
class HomomorphismResult:
    """Container for a group homomorphism's kernel and image."""

    kernel: list
    image: list
    is_homomorphism: bool


class Polynomial:
    r"""A polynomial with coefficients over :math:`\mathbb{Q}` (``modulus=None``,
    stored as exact :class:`fractions.Fraction`) or :math:`\mathrm{GF}(p)`
    (``modulus=p``, coefficients reduced mod ``p``).

    Coefficients are stored lowest-degree first: ``coeffs[k]`` is the
    coefficient of :math:`x^k`. Integer (``Z``) and rational (``Q``)
    coefficients are both represented exactly as ``Fraction`` (``Z`` is
    simply the sub-case where every coefficient happens to have
    denominator 1); this is what makes polynomial *division* with
    remainder always well-defined (needs the divisor's leading
    coefficient to be invertible, which holds automatically in a field
    but not in :math:`\mathbb{Z}` itself). See Dummit & Foote, *Abstract
    Algebra*, 3rd ed., Ch. 9.

    Parameters
    ----------
    coeffs : sequence
        Coefficients, lowest degree first.
    modulus : int, optional
        A prime ``p``, for coefficients in :math:`\mathrm{GF}(p)`;
        ``None`` (the default) for :math:`\mathbb{Q}` coefficients.

    Examples
    --------
    >>> p = Polynomial([1, 2, 1])  # 1 + 2x + x^2 = (1+x)^2
    >>> q = Polynomial([1, 1])  # 1 + x
    >>> quotient, remainder = divmod(p, q)
    >>> quotient.coeffs, remainder.coeffs
    ([Fraction(1, 1), Fraction(1, 1)], [Fraction(0, 1)])
    """

    def __init__(self, coeffs, modulus: Optional[int] = None):
        self.modulus = modulus
        if modulus is None:
            raw = [Fraction(c) for c in coeffs]
        else:
            raw = [int(c) % modulus for c in coeffs]
        self.coeffs = self._trim(raw)

    def _trim(self, coeffs: list) -> list:
        coeffs = list(coeffs)
        zero = Fraction(0) if self.modulus is None else 0
        while len(coeffs) > 1 and coeffs[-1] == zero:
            coeffs.pop()
        if not coeffs:
            coeffs = [zero]
        return coeffs

    def _reduce(self, c):
        return c if self.modulus is None else c % self.modulus

    def _wrap(self, coeffs: list) -> Polynomial:
        result = Polynomial.__new__(Polynomial)
        result.modulus = self.modulus
        result.coeffs = result._trim(coeffs)
        return result

    @property
    def degree(self) -> int:
        """int: The polynomial's degree (``-1`` for the zero polynomial, by convention)."""
        if len(self.coeffs) == 1 and self.coeffs[0] == (Fraction(0) if self.modulus is None else 0):
            return -1
        return len(self.coeffs) - 1

    def _check_same_field(self, other: Polynomial) -> None:
        """Reject arithmetic between polynomials over different coefficient fields."""
        if not isinstance(other, Polynomial):
            raise TypeError(f"expected a Polynomial, got {type(other).__name__}")
        if self.modulus != other.modulus:
            raise ValueError(f"cannot combine polynomials over different fields (modulus {self.modulus} vs {other.modulus})")

    def _pad(self, n: int) -> list:
        zero = Fraction(0) if self.modulus is None else 0
        return self.coeffs + [zero] * (n - len(self.coeffs))

    def __add__(self, other: Polynomial) -> Polynomial:
        self._check_same_field(other)
        n = max(len(self.coeffs), len(other.coeffs))
        return self._wrap([self._reduce(a + b) for a, b in zip(self._pad(n), other._pad(n))])

    def __sub__(self, other: Polynomial) -> Polynomial:
        self._check_same_field(other)
        n = max(len(self.coeffs), len(other.coeffs))
        return self._wrap([self._reduce(a - b) for a, b in zip(self._pad(n), other._pad(n))])

    def __mul__(self, other: Polynomial) -> Polynomial:
        self._check_same_field(other)
        result = [Fraction(0) if self.modulus is None else 0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = self._reduce(result[i + j] + a * b)
        return self._wrap(result)

    def __eq__(self, other) -> bool:
        return isinstance(other, Polynomial) and self.modulus == other.modulus and self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash((self.modulus, tuple(self.coeffs)))

    def __divmod__(self, other: Polynomial):
        r"""Polynomial long division: ``self = quotient * other + remainder``, ``deg(remainder) < deg(other)``."""
        self._check_same_field(other)
        if other.degree == -1:
            raise ZeroDivisionError("division by the zero polynomial")
        remainder = self._wrap(list(self.coeffs))
        quotient_degree = max(self.degree - other.degree, -1)
        zero = Fraction(0) if self.modulus is None else 0
        quotient_coeffs = [zero] * (quotient_degree + 1) if quotient_degree >= 0 else [zero]
        lead_other = other.coeffs[other.degree]
        lead_inv = (1 / lead_other) if self.modulus is None else pow(int(lead_other), self.modulus - 2, self.modulus)
        while remainder.degree >= other.degree and remainder.degree >= 0:
            shift = remainder.degree - other.degree
            factor = self._reduce(remainder.coeffs[remainder.degree] * lead_inv)
            quotient_coeffs[shift] = self._reduce(quotient_coeffs[shift] + factor)
            subtrahend_coeffs = [zero] * shift + [self._reduce(factor * c) for c in other.coeffs]
            remainder = remainder - self._wrap(subtrahend_coeffs)
        return self._wrap(quotient_coeffs), remainder

    def evaluate(self, x):
        """Evaluate the polynomial at ``x``, via Horner's method.

        Parameters
        ----------
        x

        Returns
        -------
        The same type as the polynomial's coefficients.
        """
        result = Fraction(0) if self.modulus is None else 0
        for c in reversed(self.coeffs):
            result = self._reduce(result * x + c)
        return result

    def __repr__(self) -> str:
        terms = [f"{c}*x^{k}" for k, c in enumerate(self.coeffs) if c != (Fraction(0) if self.modulus is None else 0)]
        return " + ".join(terms) if terms else "0"

"""Result containers for mathematicskit.number_theory.

Every algorithm in this domain is hand-rolled from its textbook
definition -- number theory here is exact-integer arithmetic with no
``numpy``/``scipy`` equivalent at all (unlike most other mathematicskit
domains, whose new dependency rule prefers a library call whenever one
exists). There is no natural shared ABC across this domain's algorithm
families (extended Euclid, primality, CRT, continued fractions,
Diophantine solvers are each their own self-contained function), so --
following physicskit's own precedent of varying internal shape by what a
family of algorithms actually needs -- this module holds only the small
dataclass results those functions return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = ["BezoutResult", "ContinuedFractionResult", "LinearDiophantineResult", "PellResult", "CRTResult"]


@dataclass
class BezoutResult:
    """Container for the extended Euclidean algorithm's output: Bezout's identity."""

    gcd: int
    """int: :math:`\\gcd(a, b)`."""

    x: int
    y: int
    """int: Bezout coefficients with :math:`ax + by = \\gcd(a,b)`."""


@dataclass
class ContinuedFractionResult:
    """Container for a continued-fraction expansion and its convergents."""

    terms: list[int]
    """list of int: :math:`[a_0; a_1, a_2, \\dots]`."""

    convergents: list[tuple[int, int]]
    """list of (int, int): Successive :math:`(p_k, q_k)` best rational
    approximations :math:`p_k/q_k`, in the same order as `terms`."""


@dataclass
class LinearDiophantineResult:
    """Container for a linear Diophantine equation's solution family."""

    has_solution: bool
    """bool: Whether :math:`ax + by = c` has an integer solution
    (equivalently, whether :math:`\\gcd(a,b)` divides ``c``)."""

    x0: Optional[int] = None
    y0: Optional[int] = None
    """int, optional: A particular solution, if one exists."""

    gcd: int = 0
    """int: :math:`\\gcd(a, b)`."""

    x_step: int = 0
    y_step: int = 0
    """int: The general solution is ``(x0 + k*x_step, y0 - k*y_step)``
    for any integer ``k``."""


@dataclass
class PellResult:
    """Container for the fundamental solution of Pell's equation :math:`x^2 - Dy^2 = 1`."""

    x: int
    y: int
    d: int
    """int: The ``D`` in :math:`x^2 - Dy^2 = 1`."""

    extra: dict = field(default_factory=dict)


@dataclass
class CRTResult:
    """Container for the Chinese Remainder Theorem's combined solution."""

    residue: int
    """int: The unique solution modulo `modulus`."""

    modulus: int
    """int: The combined modulus (the product of the pairwise-coprime
    input moduli)."""

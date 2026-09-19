r"""The Chinese Remainder Theorem: combining congruences with pairwise-coprime moduli.

No numpy/scipy equivalent. See Niven, Zuckerman & Montgomery, *An
Introduction to the Theory of Numbers*, 5th ed., Sec. 2.3.
"""

from __future__ import annotations

from collections.abc import Sequence

from mathkit.number_theory.core.base import CRTResult
from mathkit.number_theory.systems.modular_arithmetic import extended_gcd

__all__ = ["chinese_remainder_theorem"]


def chinese_remainder_theorem(remainders: Sequence[int], moduli: Sequence[int]) -> CRTResult:
    r"""Solve the system :math:`x \equiv r_i \pmod{m_i}` for pairwise-coprime :math:`m_i`.

    Combines the congruences two at a time: given a solution :math:`x
    \equiv r_1 \pmod{m_1}` and a new congruence :math:`x \equiv r_2
    \pmod{m_2}`, the combined solution modulo :math:`m_1 m_2` is found
    via the extended Euclidean algorithm's Bezout coefficients for
    :math:`(m_1, m_2)`. See Niven, Zuckerman & Montgomery, *An
    Introduction to the Theory of Numbers*, 5th ed., Theorem 2.18.

    Parameters
    ----------
    remainders : sequence of int
    moduli : sequence of int
        Pairwise coprime, i.e. ``gcd(moduli[i], moduli[j]) == 1`` for
        every ``i != j``.

    Returns
    -------
    CRTResult

    Examples
    --------
    >>> # x = 2 mod 3, x = 3 mod 5, x = 2 mod 7 -> x = 23 mod 105 (a classic example).
    >>> result = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    >>> result.residue
    23
    >>> result.modulus
    105
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    if len(remainders) == 0:
        raise ValueError("need at least one congruence")

    x, m = remainders[0] % moduli[0], moduli[0]
    for r_i, m_i in zip(remainders[1:], moduli[1:]):
        bezout = extended_gcd(m, m_i)
        if bezout.gcd != 1:
            raise ValueError(f"moduli must be pairwise coprime, but gcd({m}, {m_i}) = {bezout.gcd}")
        # Combined solution mod (m * m_i), via x_new = x + m * t * (r_i - x) / gcd
        # simplified using Bezout's identity m*u + m_i*v = 1.
        combined_modulus = m * m_i
        x = (x + m * bezout.x * (r_i - x)) % combined_modulus
        m = combined_modulus
    return CRTResult(residue=x, modulus=m)

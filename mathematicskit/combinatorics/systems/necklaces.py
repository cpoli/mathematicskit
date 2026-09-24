r"""Counting necklaces and bracelets with Pólya's enumeration theorem.

For the rotation group :math:`C_n`, Pólya's cycle-index count reduces to

.. math::

   N(n, k) = \frac{1}{n} \sum_{i=0}^{n-1} k^{\gcd(i, n)},

since rotating by :math:`i` positions splits the beads into
:math:`\gcd(i, n)` cycles. Adding the :math:`n` reflections of the
dihedral group gives the bracelet count. Hand-rolled (exact integers).
See G. Pólya, "Kombinatorische Anzahlbestimmungen für Gruppen, Graphen
und chemische Verbindungen," Acta Mathematica 68 (1937), 145-254.
"""

from __future__ import annotations

from math import gcd

__all__ = ["count_necklaces", "count_bracelets"]


def count_necklaces(n: int, k: int) -> int:
    r"""Number of necklaces of ``n`` beads in ``k`` colors, up to rotation.

    Parameters
    ----------
    n : int
        Number of beads, ``n >= 1``.
    k : int
        Number of colors.

    Returns
    -------
    int

    Examples
    --------
    >>> [count_necklaces(n, 2) for n in range(1, 9)]
    [2, 3, 4, 6, 8, 14, 20, 36]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return sum(k ** gcd(i, n) for i in range(n)) // n


def count_bracelets(n: int, k: int) -> int:
    r"""Number of bracelets of ``n`` beads in ``k`` colors, up to rotation and reflection.

    Parameters
    ----------
    n : int
    k : int

    Returns
    -------
    int

    Examples
    --------
    >>> [count_bracelets(n, 2) for n in range(1, 9)]
    [2, 3, 4, 6, 8, 13, 18, 30]
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rotations = sum(k ** gcd(i, n) for i in range(n))
    if n % 2:
        reflections = n * k ** ((n + 1) // 2)
    else:
        reflections = (n // 2) * (k ** (n // 2 + 1) + k ** (n // 2))
    return (rotations + reflections) // (2 * n)

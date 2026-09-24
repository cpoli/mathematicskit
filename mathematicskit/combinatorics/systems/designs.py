r"""Latin squares and orthogonal Latin squares, the combinatorial designs
behind Leonhard Euler's 1782 problem of the 36 officers.

Hand-rolled: no scipy/numpy equivalent. See C. J. Colbourn and J. H.
Dinitz, eds., *Handbook of Combinatorial Designs*, 2nd ed. (Boca Raton:
Chapman & Hall/CRC, 2007), Part III.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cyclic_latin_square", "is_latin_square", "are_orthogonal", "orthogonal_latin_square_pair"]


def cyclic_latin_square(n: int, multiplier: int = 1) -> np.ndarray:
    r"""The Latin square :math:`L_{ij} = (a\,i + j) \bmod n` for a multiplier :math:`a` coprime to ``n``.

    Parameters
    ----------
    n : int
    multiplier : int
        The coefficient :math:`a`; it must be coprime to ``n`` for every
        column to be a permutation.

    Returns
    -------
    ndarray, shape (n, n), int

    Examples
    --------
    >>> cyclic_latin_square(3)
    array([[0, 1, 2],
           [1, 2, 0],
           [2, 0, 1]])
    """
    i, j = np.indices((n, n))
    square = (multiplier * i + j) % n
    if not is_latin_square(square):
        raise ValueError(f"multiplier {multiplier} is not coprime to n = {n}")
    return square


def is_latin_square(square) -> bool:
    r"""Whether every row and every column of an :math:`n \times n` array is a permutation of ``0..n-1``.

    Parameters
    ----------
    square : array_like, shape (n, n)

    Returns
    -------
    bool

    Examples
    --------
    >>> is_latin_square([[0, 1], [1, 0]]), is_latin_square([[0, 1], [0, 1]])
    (True, False)
    """
    square = np.asarray(square)
    n = square.shape[0]
    target = np.arange(n)
    return square.shape == (n, n) and all(np.array_equal(np.sort(line), target) for line in (*square, *square.T))


def are_orthogonal(a, b) -> bool:
    r"""Whether two Latin squares are orthogonal: superimposed, every ordered pair of symbols appears exactly once.

    Parameters
    ----------
    a, b : array_like, shape (n, n)

    Returns
    -------
    bool

    Examples
    --------
    >>> are_orthogonal(cyclic_latin_square(3, 1), cyclic_latin_square(3, 2))
    True
    """
    a, b = np.asarray(a), np.asarray(b)
    n = a.shape[0]
    return len(set(zip(a.ravel().tolist(), b.ravel().tolist()))) == n * n


def orthogonal_latin_square_pair(n: int):
    r"""A pair of orthogonal Latin squares of odd order ``n``: :math:`L_{ij} = i + j` and :math:`M_{ij} = 2i + j \pmod n`.

    Euler conjectured in 1782 that no pair exists when
    :math:`n \equiv 2 \pmod 4`. Gaston Tarry confirmed it for
    :math:`n = 6` in 1900, but R. C. Bose, S. S. Shrikhande, and E. T.
    Parker disproved the conjecture for every other such :math:`n > 6`
    in 1959-1960. This construction covers odd :math:`n`, where 2 is
    invertible modulo :math:`n`.

    Parameters
    ----------
    n : int
        Odd, ``n >= 3``.

    Returns
    -------
    tuple of ndarray

    Examples
    --------
    >>> a, b = orthogonal_latin_square_pair(5)
    >>> are_orthogonal(a, b)
    True
    """
    if n < 3 or n % 2 == 0:
        raise ValueError("this construction needs odd n >= 3")
    return cyclic_latin_square(n, 1), cyclic_latin_square(n, 2)

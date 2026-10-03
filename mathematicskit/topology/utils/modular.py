"""Matrix rank over a prime field GF(p), for Betti numbers with mod-p coefficients.

Hand-rolled: numpy's ``matrix_rank`` works in floating point, over the
reals, so it cannot see that a boundary matrix drops rank mod 2 (which
is how torsion shows up in mod-p homology). Over GF(2) each column is
packed into a Python integer used as a bit set, so eliminating a column
is one XOR; this is fast enough for the tens of thousands of simplices
of a Vietoris-Rips complex.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rank_mod_p"]


def rank_mod_p(A, p: int) -> int:
    """Rank of an integer matrix over the field GF(p), by Gaussian elimination.

    Parameters
    ----------
    A : array_like of int, shape (m, n)
    p : int
        A prime.

    Returns
    -------
    int

    Examples
    --------
    >>> rank_mod_p([[2, 0], [0, 1]], 2), rank_mod_p([[2, 0], [0, 1]], 3)
    (1, 2)
    """
    M = np.array(A, dtype=np.int64) % p
    if M.ndim != 2 or M.size == 0:
        return 0
    if p == 2:
        return _rank_gf2(M)
    rows, cols = M.shape
    rank = 0
    for c in range(cols):
        if rank == rows:
            break
        nonzero = np.nonzero(M[rank:, c])[0]
        if nonzero.size == 0:
            continue
        pivot = rank + nonzero[0]
        M[[rank, pivot]] = M[[pivot, rank]]
        M[rank] = (M[rank] * pow(int(M[rank, c]), -1, p)) % p
        below = rank + 1 + np.nonzero(M[rank + 1 :, c])[0]
        M[below] = (M[below] - np.outer(M[below, c], M[rank])) % p
        rank += 1
    return rank


def _rank_gf2(M: np.ndarray) -> int:
    weights = [1 << i for i in range(M.shape[0])]
    pivots: dict[int, int] = {}
    for column in M.T:
        bits = sum(weights[i] for i in np.flatnonzero(column))
        while bits:
            low = bits.bit_length() - 1
            if low not in pivots:
                pivots[low] = bits
                break
            bits ^= pivots[low]
    return len(pivots)

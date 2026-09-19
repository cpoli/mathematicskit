r"""Pascal's triangle: the binomial-coefficient recurrence, built by hand.

Kept hand-rolled purely as a pedagogical illustration of the recurrence
:math:`\binom{n}{k} = \binom{n-1}{k-1}+\binom{n-1}{k}` itself -- for
actually *computing* a binomial coefficient,
:func:`mathematicskit.combinatorics.systems.counting.combinations_count`
(:func:`scipy.special.comb`) is the primary API. See Graham, Knuth &
Patashnik, *Concrete Mathematics*, 2nd ed., Sec. 5.1, eq. (5.8) (the
"Pascal's Triangle" recurrence and its addition-formula proof).
"""

from __future__ import annotations

__all__ = ["pascals_triangle"]


def pascals_triangle(n_rows: int) -> list:
    r"""Build the first `n_rows` rows of Pascal's triangle via the addition recurrence.

    Row ``n`` (0-indexed) holds :math:`\binom{n}{0}, \dots, \binom{n}{n}`,
    each computed as the sum of the two entries above it in the previous
    row (with implicit zeros outside the triangle) -- never calling
    :func:`~mathematicskit.combinatorics.systems.counting.combinations_count`
    at all. See Graham, Knuth & Patashnik, *Concrete Mathematics*, 2nd
    ed., Sec. 5.1.

    Parameters
    ----------
    n_rows : int

    Returns
    -------
    list of list of int

    Examples
    --------
    >>> pascals_triangle(5)
    [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    """
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    triangle = [[1]]
    for n in range(1, n_rows):
        prev = triangle[-1]
        row = [1] + [prev[k - 1] + prev[k] for k in range(1, n)] + [1]
        triangle.append(row)
    return triangle

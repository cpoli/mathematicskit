r"""Lattice basis reduction: the Lenstra-Lenstra-Lovász (LLL) algorithm.

A lattice :math:`L = \{\sum_i k_i b_i : k_i \in \mathbb Z\}` has many
bases, most of them long and nearly parallel. LLL (1982) finds, in
polynomial time, a basis of short, nearly orthogonal vectors. With
Gram-Schmidt vectors :math:`b_i^*` and coefficients
:math:`\mu_{ij} = \langle b_i, b_j^* \rangle / \|b_j^*\|^2`, the basis is
*reduced* when

.. math::

   |\mu_{ij}| \leq \tfrac12 \ (j < i) \qquad\text{and}\qquad
   \|b_k^*\|^2 \geq (\delta - \mu_{k,k-1}^2)\,\|b_{k-1}^*\|^2,

the size condition and Lovász's condition, with
:math:`\tfrac14 < \delta < 1`. Its first vector is then within a factor
:math:`2^{(n-1)/2}` of the shortest nonzero lattice vector (for
:math:`\delta = 3/4`). The same algorithm factors polynomials over
:math:`\mathbb Q`, breaks knapsack cryptosystems, and finds integer
relations among real numbers. Hand-rolled in exact rational arithmetic
(:class:`fractions.Fraction`), so rounding never corrupts the
reduction. See A. K. Lenstra, H. W. Lenstra Jr. and L. Lovász,
"Factoring Polynomials with Rational Coefficients," Mathematische
Annalen 261 (1982), 515-534, and Cohen, *A Course in Computational
Algebraic Number Theory* (1993), Sec. 2.6.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from mathematicskit.number_theory.core.base import LLLResult

__all__ = ["lll_reduce", "integer_relation"]


def _dot(u, v) -> Fraction:
    return sum((Fraction(a) * b for a, b in zip(u, v, strict=True)), Fraction(0))


def _gram_schmidt(basis: list) -> tuple:
    n = len(basis)
    ortho: list = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    norms: list = []
    for i, b in enumerate(basis):
        v = [Fraction(c) for c in b]
        for j in range(i):
            mu[i][j] = _dot(b, ortho[j]) / norms[j] if norms[j] else Fraction(0)
            v = [vc - mu[i][j] * oc for vc, oc in zip(v, ortho[j], strict=True)]
        ortho.append(v)
        norms.append(_dot(v, v))
    return mu, norms


def lll_reduce(basis: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> LLLResult:
    r"""LLL-reduce a lattice basis given by linearly independent integer rows.

    Works through the basis left to right: size-reduce :math:`b_k`
    against every earlier vector (subtract :math:`\lfloor\mu_{kj}\rceil b_j`),
    then either advance, if Lovász's condition holds, or swap
    :math:`b_k` with :math:`b_{k-1}` and step back. Each swap shrinks the
    product :math:`\prod_i \|b_i^*\|^{2(n-i)}` by at least a factor
    :math:`\delta`, which bounds the number of swaps.

    Parameters
    ----------
    basis : sequence of sequence of int
        Basis vectors as rows (``n`` vectors in :math:`\mathbb Z^m`, ``n <= m``).
    delta : Fraction
        Lovász parameter in ``(1/4, 1)``; closer to 1 gives a better
        reduction for more work.

    Returns
    -------
    LLLResult

    Examples
    --------
    >>> lll_reduce([[1, 1, 1], [-1, 0, 2], [3, 5, 6]]).basis
    [[0, 1, 0], [1, 0, 1], [-1, 0, 2]]
    >>> lll_reduce([[201, 37], [1648, 297]]).basis  # Hoffstein, Pipher and Silverman's example
    [[1, 32], [40, 1]]
    """
    b = [[int(c) for c in row] for row in basis]
    n = len(b)
    delta = Fraction(delta)
    mu, norms = _gram_schmidt(b)
    if any(norm == 0 for norm in norms):
        raise ValueError("basis vectors must be linearly independent")
    swaps = 0
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                b[k] = [x - q * y for x, y in zip(b[k], b[j], strict=True)]
                for i in range(j + 1):
                    mu[k][i] -= q * (mu[j][i] if i < j else 1)
        if norms[k] >= (delta - mu[k][k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            swaps += 1
            mu, norms = _gram_schmidt(b)
            k = max(k - 1, 1)
    return LLLResult(basis=b, swaps=swaps, gram_schmidt_norms=[float(x) for x in norms])


def integer_relation(values: Sequence[float], scale: float = 1e12) -> list:
    r"""Small integers :math:`a_i`, not all zero, with :math:`\sum_i a_i x_i \approx 0`.

    Reduces the lattice spanned by the rows of
    :math:`[\,I_n \mid \lfloor C x\rceil\,]` (Lenstra, Lenstra and Lovász's
    own application, in their paper's Sec. 1): a short vector
    :math:`(a, \sum_i a_i \lfloor C x_i \rceil)` needs both small
    coefficients and a nearly vanishing combination. With
    :math:`x = (1, \alpha, \dots, \alpha^d)` this recovers the minimal
    polynomial of an algebraic number :math:`\alpha` from its decimal
    expansion. The scale :math:`C` should be well below
    :math:`1/\text{(precision of } x)`.

    Parameters
    ----------
    values : sequence of float
    scale : float

    Returns
    -------
    list of int
        The coefficients, from the first vector of the reduced basis.

    Examples
    --------
    >>> import math
    >>> alpha = math.sqrt(2) + math.sqrt(3)
    >>> integer_relation([alpha**k for k in range(5)], scale=1e10)  # x^4 - 10x^2 + 1
    [1, 0, -10, 0, 1]
    """
    n = len(values)
    rows = [[int(i == j) for j in range(n)] + [round(scale * x)] for i, x in enumerate(values)]
    coefficients = lll_reduce(rows).basis[0][:n]
    for c in coefficients:  # normalize the sign: first nonzero coefficient positive
        if c:
            return coefficients if c > 0 else [-c for c in coefficients]
    return coefficients

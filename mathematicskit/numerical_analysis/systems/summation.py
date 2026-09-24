"""Kahan's compensated summation.

See W. Kahan, "Pracniques: Further remarks on reducing truncation
errors," Communications of the ACM 8(1) (1965), 40; N. J. Higham, "The
accuracy of floating point summation," SIAM Journal on Scientific
Computing 14 (1993), 783-799. Neither numpy nor scipy offers Kahan's
algorithm (``numpy.sum`` uses pairwise summation, and ``math.fsum`` an
exact but different algorithm), and the loop itself is the subject here,
so it is hand-written. (Python 3.12+'s built-in ``sum`` itself uses a
compensated variant for floats, so naive summation must be written as an
explicit loop to see the rounding it avoids.)
"""

from __future__ import annotations

import numpy as np

__all__ = ["kahan_sum"]


def kahan_sum(values) -> float:
    r"""Sum floating-point numbers with Kahan's compensated summation.

    A running correction ``c`` captures the low-order bits lost when each
    term is added to the (much larger) partial sum, and feeds them back
    into the next term:

    .. code-block:: text

        y = x_i - c
        t = s + y
        c = (t - s) - y
        s = t

    The computed sum satisfies
    :math:`|\hat s - s| \le \bigl(2u + O(nu^2)\bigr) \sum |x_i|`, with
    :math:`u` the unit roundoff, independent of :math:`n` to first order,
    where naive left-to-right summation only guarantees
    :math:`(n - 1)\,u \sum |x_i|` (Higham 1993).

    Parameters
    ----------
    values : array-like of float
        The terms to add.

    Returns
    -------
    float

    Examples
    --------
    >>> values = [1.0] + [1e-16] * 10_000
    >>> naive = 0.0
    >>> for v in values:  # left to right: every tiny term is rounded away
    ...     naive += v
    >>> naive
    1.0
    >>> round(kahan_sum(values), 15)
    1.000000000001
    """
    s = 0.0
    c = 0.0
    for x in np.asarray(values, dtype=np.float64).ravel().tolist():
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

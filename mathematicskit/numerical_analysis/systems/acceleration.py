"""Convergence acceleration of sequences: Aitken's :math:`\\Delta^2` process.

See A. C. Aitken, "On Bernoulli's numerical solution of algebraic
equations," Proceedings of the Royal Society of Edinburgh 46 (1926),
289-305; Burden & Faires, *Numerical Analysis*, 10th ed., Ch. 2.5.
Neither numpy nor scipy exposes the transform on a plain sequence
(:func:`scipy.optimize.fixed_point` uses it only inside its own
iteration), so it is written out here; it is a few vectorized differences.
"""

from __future__ import annotations

import numpy as np

__all__ = ["aitken_delta_squared"]


def aitken_delta_squared(sequence) -> np.ndarray:
    r"""Apply Aitken's :math:`\Delta^2` process to a sequence.

    .. math::

        \hat a_n = a_n - \frac{(\Delta a_n)^2}{\Delta^2 a_n}
                 = a_n - \frac{(a_{n+1} - a_n)^2}{a_{n+2} - 2 a_{n+1} + a_n}

    The transform is exact for sequences of the form
    :math:`a_n = L + c\,r^n` (:math:`r \neq 1`), and for any linearly
    convergent sequence it converges to the limit faster than the
    original. Where :math:`\Delta^2 a_n = 0`, :math:`a_{n+2}` is returned.

    Parameters
    ----------
    sequence : array-like, shape (n,)
        Terms :math:`a_0, \ldots, a_{n-1}`, with ``n >= 3``.

    Returns
    -------
    ndarray, shape (n - 2,)
        The accelerated sequence :math:`\hat a_0, \ldots, \hat a_{n-3}`.

    Examples
    --------
    >>> import numpy as np
    >>> # a_n = 2 + 3 * 0.5**n: one Aitken step recovers the limit exactly.
    >>> a = 2.0 + 3.0 * 0.5 ** np.arange(6)
    >>> np.allclose(aitken_delta_squared(a), 2.0)
    True
    """
    a = np.asarray(sequence, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < 3:
        raise ValueError("sequence must be 1-D with at least 3 terms")
    d1 = a[1:-1] - a[:-2]
    d2 = a[2:] - 2.0 * a[1:-1] + a[:-2]
    out = a[2:].copy()
    nz = d2 != 0.0
    out[nz] = a[:-2][nz] - d1[nz] ** 2 / d2[nz]
    return out

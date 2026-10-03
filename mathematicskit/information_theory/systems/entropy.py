r"""Measures of information: Hartley's logarithm, Shannon entropy, joint and
conditional entropy, mutual information, and the Kullback-Leibler divergence.

Thin wrappers over :func:`scipy.stats.entropy` and
:func:`scipy.special.entr`, which already compute
:math:`-\sum p\log p` and :math:`\sum p\log(p/q)` with the
:math:`0\log 0 = 0` convention. See T. M. Cover and J. A. Thomas,
*Elements of Information Theory*, 2nd ed. (Wiley, 2006), Ch. 2.
"""

from __future__ import annotations

import numpy as np
from scipy.special import entr
from scipy.stats import entropy as _scipy_entropy

__all__ = [
    "hartley_information",
    "entropy",
    "binary_entropy",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "kl_divergence",
]


def _distribution(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p < 0):
        raise ValueError("probabilities must be non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("probabilities must not all be zero")
    return p / total


def hartley_information(n_messages, length: int = 1, base: float = 2) -> float:
    r"""Hartley's measure :math:`H_0 = \text{length}\cdot\log(n)` of a choice among equally likely messages.

    R. V. L. Hartley, "Transmission of Information," Bell System Technical
    Journal 7(3) (1928), 535-563: a sequence of `length` selections from
    an alphabet of `n_messages` symbols carries :math:`\text{length}\log n`
    units of information.

    Parameters
    ----------
    n_messages : int or array_like
        Alphabet size :math:`n \ge 1`.
    length : int
        Number of selections.
    base : float
        Logarithm base: 2 gives bits, 10 gives hartleys.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> round(hartley_information(26), 4)  # one letter
    4.7004
    >>> hartley_information(2, length=8)  # one byte
    8.0
    """
    n = np.asarray(n_messages, dtype=float)
    if np.any(n < 1):
        raise ValueError("n_messages must be at least 1")
    result = length * np.log(n) / np.log(base)
    return float(result) if result.ndim == 0 else result


def entropy(p, base: float = 2) -> float:
    r"""Shannon entropy :math:`H(X) = -\sum_x p(x)\log p(x)`.

    C. E. Shannon, "A Mathematical Theory of Communication," Bell System
    Technical Journal 27 (1948), 379-423, Theorem 2. Computed by
    :func:`scipy.stats.entropy`.

    Parameters
    ----------
    p : array_like
        Probabilities (normalized if they don't sum to 1). A 2-D array is
        treated as one joint distribution.
    base : float
        Logarithm base; 2 gives bits.

    Returns
    -------
    float

    Examples
    --------
    >>> entropy([0.5, 0.25, 0.25])
    1.5
    >>> entropy([1, 0, 0])
    0.0
    """
    return float(_scipy_entropy(_distribution(p).ravel(), base=base))


def binary_entropy(p):
    r"""The binary entropy function :math:`h(p) = -p\log_2 p - (1-p)\log_2(1-p)`.

    Parameters
    ----------
    p : float or array_like
        Probability in :math:`[0, 1]`.

    Returns
    -------
    float or ndarray
        In bits.

    Examples
    --------
    >>> binary_entropy(0.5)
    1.0
    >>> round(binary_entropy(0.11), 4)
    0.4999
    """
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must lie in [0, 1]")
    h = (entr(p) + entr(1.0 - p)) / np.log(2.0)
    return float(h) if h.ndim == 0 else h


def joint_entropy(joint, base: float = 2) -> float:
    r"""Joint entropy :math:`H(X,Y)` of a joint distribution ``joint[i, j] = P(X=i, Y=j)``.

    Parameters
    ----------
    joint : array_like, shape (m, n)
    base : float

    Returns
    -------
    float

    Examples
    --------
    >>> joint_entropy([[0.25, 0.25], [0.25, 0.25]])
    2.0
    """
    return entropy(joint, base=base)


def conditional_entropy(joint, base: float = 2) -> float:
    r"""Conditional entropy :math:`H(Y\mid X) = H(X,Y) - H(X)`, with :math:`X` indexing the rows.

    Parameters
    ----------
    joint : array_like, shape (m, n)
        ``joint[i, j] = P(X=i, Y=j)``.
    base : float

    Returns
    -------
    float

    Examples
    --------
    >>> conditional_entropy([[0.5, 0.0], [0.0, 0.5]])  # Y = X
    0.0
    """
    joint = _distribution(joint)
    return float(max(entropy(joint, base) - entropy(joint.sum(axis=1), base), 0.0))


def mutual_information(joint, base: float = 2) -> float:
    r"""Mutual information :math:`I(X;Y) = H(X) + H(Y) - H(X,Y)`.

    Shannon (1948), Section 20: the rate of transmission of a noisy
    channel is :math:`H(X) - H(X\mid Y)`. It is zero exactly when
    :math:`X` and :math:`Y` are independent.

    Parameters
    ----------
    joint : array_like, shape (m, n)
        ``joint[i, j] = P(X=i, Y=j)``.
    base : float

    Returns
    -------
    float

    Examples
    --------
    >>> mutual_information([[0.5, 0.0], [0.0, 0.5]])
    1.0
    >>> mutual_information(np.outer([0.3, 0.7], [0.6, 0.4]))  # independent
    0.0
    """
    joint = _distribution(joint)
    value = entropy(joint.sum(axis=1), base) + entropy(joint.sum(axis=0), base) - entropy(joint, base)
    return float(max(round(value, 15), 0.0))


def kl_divergence(p, q, base: float = 2) -> float:
    r"""Kullback-Leibler divergence :math:`D(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)}`.

    S. Kullback and R. A. Leibler, "On Information and Sufficiency,"
    Annals of Mathematical Statistics 22(1) (1951), 79-86. Non-negative
    (Gibbs' inequality), zero only when :math:`p = q`, and not symmetric.
    Infinite when :math:`q(x) = 0 < p(x)`. Computed by
    :func:`scipy.stats.entropy`.

    Parameters
    ----------
    p, q : array_like
        Distributions on the same alphabet.
    base : float

    Returns
    -------
    float

    Examples
    --------
    >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> round(kl_divergence([0.5, 0.5], [0.9, 0.1]), 4), round(kl_divergence([0.9, 0.1], [0.5, 0.5]), 4)
    (0.737, 0.531)
    """
    p, q = _distribution(p), _distribution(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    return float(_scipy_entropy(p, q, base=base))

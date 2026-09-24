r"""Continuous-time Markov chains through Kolmogorov's differential equations.

A continuous-time chain on a finite state space is described by a
generator (rate) matrix :math:`Q`: off-diagonal entries are jump rates,
and each row sums to zero. Kolmogorov (1931) showed that the transition
matrix :math:`P(t)` solves the forward and backward equations
:math:`P'(t) = P(t)Q = QP(t)`, :math:`P(0) = I`, so
:math:`P(t) = e^{tQ}`, evaluated here with :func:`scipy.linalg.expm`.
See Norris, *Markov Chains* (1997), Ch. 2-3.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

__all__ = ["ctmc_transition_matrix", "ctmc_stationary_distribution"]


def _check_generator(generator) -> np.ndarray:
    Q = np.asarray(generator, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("generator must be a square matrix.")
    off_diagonal = Q[~np.eye(Q.shape[0], dtype=bool)]
    if np.any(off_diagonal < 0) or not np.allclose(Q.sum(axis=1), 0.0):
        raise ValueError("generator must have non-negative off-diagonal rates and rows summing to zero.")
    return Q


def ctmc_transition_matrix(generator, t: float) -> np.ndarray:
    r"""Transition matrix :math:`P(t) = e^{tQ}` of a continuous-time Markov chain.

    Parameters
    ----------
    generator : array-like, shape (n, n)
        Generator matrix :math:`Q`.
    t : float
        Elapsed time, ``t >= 0``.

    Returns
    -------
    ndarray, shape (n, n)
        ``P[i, j]`` is the probability of being in state ``j`` at time
        ``t`` having started in state ``i``.

    Examples
    --------
    >>> # Two states, rate 1 from 0 to 1 and rate 3 back: P_01(t) = (1 - e^{-4t}) / 4.
    >>> P = ctmc_transition_matrix([[-1.0, 1.0], [3.0, -3.0]], t=0.5)
    >>> bool(np.isclose(P[0, 1], (1 - np.exp(-2.0)) / 4))
    True
    """
    return linalg.expm(t * _check_generator(generator))


def ctmc_stationary_distribution(generator) -> np.ndarray:
    r"""Stationary distribution :math:`\pi` of an irreducible chain, solving :math:`\pi Q = 0`, :math:`\sum_i \pi_i = 1`.

    Solved as a least-squares system with the normalization appended to
    :math:`Q^T` (:func:`numpy.linalg.lstsq`); the solution is exact for an
    irreducible chain.

    Parameters
    ----------
    generator : array-like, shape (n, n)

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> ctmc_stationary_distribution([[-1.0, 1.0], [3.0, -3.0]]).round(4)
    array([0.75, 0.25])
    """
    Q = _check_generator(generator)
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]

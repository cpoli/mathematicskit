r"""Discrete-time Markov chains: transition matrices, stationary
distributions, and absorption probabilities.

The stationary distribution is found via :func:`numpy.linalg.eig` (the
left eigenvector for eigenvalue 1) with a hand-rolled power-iteration
alternative kept alongside it (the iteration itself -- watching
:math:`\pi_k = \pi_0 P^k` converge -- is the standard pedagogical way to
*see* a Markov chain "forget" its initial distribution, with no direct
library equivalent for that iterative process); absorption probabilities
and expected time to absorption are solved via :func:`numpy.linalg.solve`.
See Grinstead & Snell, *Introduction to Probability*, 2nd ed., Ch. 11.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["MarkovChain"]


class MarkovChain:
    r"""A discrete-time, finite-state Markov chain given its transition matrix.

    Parameters
    ----------
    transition_matrix : ndarray, shape (n, n)
        Row-stochastic matrix: ``transition_matrix[i, j]`` is
        :math:`P(X_{t+1}=j \mid X_t=i)`, so each row sums to 1.

    Examples
    --------
    >>> import numpy as np
    >>> # A simple 2-state weather chain: sunny/rainy.
    >>> P = np.array([[0.9, 0.1], [0.5, 0.5]])
    >>> chain = MarkovChain(P)
    >>> pi = chain.stationary_distribution()
    >>> np.allclose(pi, [5.0 / 6.0, 1.0 / 6.0], atol=1e-8)
    True
    """

    def __init__(self, transition_matrix: np.ndarray):
        p = np.asarray(transition_matrix, dtype=np.float64)
        n = p.shape[0]
        if p.shape != (n, n):
            raise ValueError("transition_matrix must be square")
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("every row of transition_matrix must sum to 1")
        if np.any(p < -1e-12):
            raise ValueError("transition_matrix must be non-negative")
        self.p = p
        self.n = n

    def stationary_distribution(self) -> np.ndarray:
        r"""Stationary distribution :math:`\pi` with :math:`\pi P = \pi`, :math:`\sum_i \pi_i = 1`.

        :math:`\pi` is the left eigenvector of ``P`` for eigenvalue 1,
        equivalently the right eigenvector of :math:`P^T` -- computed via
        :func:`numpy.linalg.eig`. Only well-defined (unique) for an
        irreducible chain; the eigenvalue closest to 1 is used, which for
        a reducible/periodic chain may not be exactly 1. See Grinstead &
        Snell, *Introduction to Probability*, 2nd ed., Theorem 11.6.

        Returns
        -------
        ndarray, shape (n,)
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.p.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        vec = np.real(eigenvectors[:, idx])
        total = float(np.sum(vec))
        # Dividing by the sum both normalizes and fixes the sign, since a
        # stationary eigenvector's entries all share one sign. A sum of
        # (near) zero means the chosen eigenvector is not a distribution at
        # all -- the chain is reducible or periodic, so no unique stationary
        # distribution exists -- and normalizing would silently return noise.
        if abs(total) < 1e-12:
            raise np.linalg.LinAlgError(
                "no unique stationary distribution: the eigenvector for eigenvalue 1 sums to zero, which means the chain is reducible or periodic"
            )
        return vec / total

    def stationary_distribution_power_iteration(self, n_iter: int = 10000, tol: float = 1e-14) -> np.ndarray:
        r"""Stationary distribution via power iteration: :math:`\pi_{k+1} = \pi_k P`.

        Starting from the uniform distribution, repeatedly applies the
        transition matrix; for an irreducible, aperiodic ("ergodic")
        chain this converges to the same stationary :math:`\pi` found by
        :meth:`stationary_distribution`, illustrating the chain
        "forgetting" its start. See Grinstead & Snell, *Introduction to
        Probability*, 2nd ed., Theorem 11.7.

        Parameters
        ----------
        n_iter : int
        tol : float
            Convergence tolerance on the total-variation change per step.

        Returns
        -------
        ndarray, shape (n,)

        Examples
        --------
        >>> import numpy as np
        >>> P = np.array([[0.9, 0.1], [0.5, 0.5]])
        >>> chain = MarkovChain(P)
        >>> pi = chain.stationary_distribution_power_iteration()
        >>> np.allclose(pi, [5.0 / 6.0, 1.0 / 6.0], atol=1e-6)
        True
        """
        pi = np.ones(self.n) / self.n
        for _ in range(n_iter):
            pi_new = pi @ self.p
            if np.sum(np.abs(pi_new - pi)) < tol:
                pi = pi_new
                break
            pi = pi_new
        return pi

    def absorption_probabilities(self, transient: Sequence[int], absorbing: Sequence[int]) -> np.ndarray:
        r"""Probability of ending in each absorbing state, from each transient state.

        Partitioning ``P`` (after reordering) into transient-to-transient
        block :math:`Q` and transient-to-absorbing block :math:`R`, the
        absorption probabilities are :math:`B = (I - Q)^{-1} R`, solved
        via :func:`numpy.linalg.solve` (never an explicit matrix
        inverse). See Grinstead & Snell, *Introduction to Probability*,
        2nd ed., Theorem 11.16.

        Parameters
        ----------
        transient : sequence of int
            Indices of the transient states.
        absorbing : sequence of int
            Indices of the absorbing states (rows with a single 1 on the
            diagonal).

        Returns
        -------
        ndarray, shape (len(transient), len(absorbing))
            ``B[i, j]``: probability of eventual absorption in
            ``absorbing[j]``, starting from ``transient[i]``.

        Examples
        --------
        >>> import numpy as np
        >>> # Gambler's ruin, capital 1..3, absorbing at 0 and 4, fair coin (p=0.5).
        >>> P = np.array([
        ...     [1.0, 0.0, 0.0, 0.0, 0.0],
        ...     [0.5, 0.0, 0.5, 0.0, 0.0],
        ...     [0.0, 0.5, 0.0, 0.5, 0.0],
        ...     [0.0, 0.0, 0.5, 0.0, 0.5],
        ...     [0.0, 0.0, 0.0, 0.0, 1.0],
        ... ])
        >>> chain = MarkovChain(P)
        >>> B = chain.absorption_probabilities(transient=[1, 2, 3], absorbing=[0, 4])
        >>> np.allclose(B[:, 1], [0.25, 0.5, 0.75], atol=1e-8)  # P(reach 4 | start at i) = i/4
        True
        """
        transient = np.asarray(transient, dtype=np.int64)
        absorbing = np.asarray(absorbing, dtype=np.int64)
        q = self.p[np.ix_(transient, transient)]
        r = self.p[np.ix_(transient, absorbing)]
        return np.linalg.solve(np.eye(transient.shape[0]) - q, r)

    def expected_steps_to_absorption(self, transient: Sequence[int]) -> np.ndarray:
        r"""Expected number of steps before absorption, from each transient state.

        :math:`t = (I-Q)^{-1}\mathbf{1}`, via :func:`numpy.linalg.solve`.
        See Grinstead & Snell, *Introduction to Probability*, 2nd ed.,
        Theorem 11.14.

        Parameters
        ----------
        transient : sequence of int
            Indices of the transient states.

        Returns
        -------
        ndarray, shape (len(transient),)

        Examples
        --------
        >>> import numpy as np
        >>> P = np.array([
        ...     [1.0, 0.0, 0.0, 0.0, 0.0],
        ...     [0.5, 0.0, 0.5, 0.0, 0.0],
        ...     [0.0, 0.5, 0.0, 0.5, 0.0],
        ...     [0.0, 0.0, 0.5, 0.0, 0.5],
        ...     [0.0, 0.0, 0.0, 0.0, 1.0],
        ... ])
        >>> chain = MarkovChain(P)
        >>> t = chain.expected_steps_to_absorption(transient=[1, 2, 3])
        >>> np.allclose(t, [3.0, 4.0, 3.0], atol=1e-8)  # i * (4 - i) for i=1,2,3
        True
        """
        transient = np.asarray(transient, dtype=np.int64)
        q = self.p[np.ix_(transient, transient)]
        ones = np.ones(transient.shape[0])
        return np.linalg.solve(np.eye(transient.shape[0]) - q, ones)

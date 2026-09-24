r"""Galton-Watson branching processes: extinction probability and simulation.

Each individual independently has :math:`j` children with probability
:math:`p_j`. With probability generating function
:math:`G(s) = \sum_j p_j s^j`, the probability :math:`q_n` that the
family is extinct by generation :math:`n` obeys :math:`q_{n+1} = G(q_n)`,
:math:`q_0 = 0`, and increases to the smallest root of :math:`G(q) = q`
in :math:`[0, 1]`. That root is 1 exactly when the mean offspring
:math:`m = G'(1) \leq 1` (unless :math:`p_1 = 1`). No scipy routine
covers branching processes, so the fixed-point iteration is written out
here; the simulation draws offspring counts with
:meth:`numpy.random.Generator.multinomial`. See Athreya & Ney,
*Branching Processes* (1972), Ch. I.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.constants import DEFAULT_ATOL, DEFAULT_MAX_ITER
from mathematicskit.probability.core.base import BranchingProcessResult

__all__ = ["galton_watson_extinction_probability", "galton_watson_simulate"]


def _check_pmf(offspring_pmf) -> np.ndarray:
    p = np.asarray(offspring_pmf, dtype=np.float64)
    if p.ndim != 1 or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("offspring_pmf must be a 1D array of non-negative probabilities summing to 1.")
    return p


def galton_watson_extinction_probability(offspring_pmf, tol: float = DEFAULT_ATOL, max_iter: int = DEFAULT_MAX_ITER) -> float:
    r"""Probability that a Galton-Watson family started by one individual eventually dies out.

    Iterates :math:`q_{n+1} = G(q_n)` from :math:`q_0 = 0`; each iterate
    is the probability of extinction by generation :math:`n`. In the
    subcritical and critical cases (:math:`m \leq 1`) the answer is 1 by
    the Galton-Watson theorem and is returned directly, since the
    critical iteration converges only like :math:`1/n`.

    Parameters
    ----------
    offspring_pmf : array-like of float
        ``offspring_pmf[j]`` is the probability of exactly ``j`` children.
    tol : float
        Stop when successive iterates differ by less than ``tol``.
    max_iter : int

    Returns
    -------
    float
        The extinction probability :math:`q`.

    Examples
    --------
    >>> # G(s) = 1/4 + s/4 + s^2/2, whose smallest fixed point is 1/2.
    >>> round(galton_watson_extinction_probability([0.25, 0.25, 0.5]), 8)
    0.5
    """
    p = _check_pmf(offspring_pmf)
    mean = float(np.dot(np.arange(p.size), p))
    if p.size > 1 and np.isclose(p[1], 1.0):
        return 0.0
    if mean <= 1.0:
        return 1.0
    q = 0.0
    for _ in range(max_iter):
        q_next = float(np.polynomial.polynomial.polyval(q, p))
        if abs(q_next - q) < tol:
            return q_next
        q = q_next
    return q


def galton_watson_simulate(offspring_pmf, n_generations: int, n_runs: int = 1000, founders: int = 1, seed: int = 0) -> BranchingProcessResult:
    r"""Simulate independent Galton-Watson processes.

    The children of the :math:`Z_n` individuals in generation :math:`n`
    are counted with one multinomial draw of size :math:`Z_n` over the
    offspring classes, so :math:`Z_{n+1} = \sum_j j\,N_j`.

    Parameters
    ----------
    offspring_pmf : array-like of float
    n_generations : int
    n_runs : int
        Number of independent families.
    founders : int
        Population of generation 0.
    seed : int

    Returns
    -------
    BranchingProcessResult

    Examples
    --------
    >>> result = galton_watson_simulate([0.25, 0.25, 0.5], n_generations=30, n_runs=4000, seed=0)
    >>> bool(abs(result.extinct_fraction - 0.5) < 0.03)
    True
    """
    p = _check_pmf(offspring_pmf)
    rng = np.random.default_rng(seed)
    j = np.arange(p.size)
    sizes = np.zeros((n_runs, n_generations + 1), dtype=np.int64)
    sizes[:, 0] = founders
    for n in range(n_generations):
        counts = rng.multinomial(sizes[:, n], p)
        sizes[:, n + 1] = counts @ j
    return BranchingProcessResult(generation_sizes=sizes, extinct_fraction=float(np.mean(sizes[:, -1] == 0)))

r"""Random walks and Brownian motion.

A simple random walk on :math:`\mathbb{Z}^d` steps to one of its
:math:`2d` neighbours uniformly at random. Pólya (1921) proved it
returns to the origin with probability 1 when :math:`d \leq 2` and with
probability less than 1 when :math:`d \geq 3`. Brownian motion is its
continuous scaling limit: :math:`W(0) = 0` with independent Gaussian
increments :math:`W(t) - W(s) \sim \mathcal N(0, t - s)`, sampled here
by cumulatively summing :func:`numpy.random.Generator.standard_normal`
increments. See Lawler & Limic, *Random Walk: A Modern Introduction*
(2010), Ch. 1 and 4, and Mörters & Peres, *Brownian Motion* (2010),
Ch. 1.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from mathematicskit.probability.core.base import BrownianMotionResult

__all__ = ["brownian_motion", "simple_random_walk", "random_walk_return_fraction", "return_probability_1d"]


def brownian_motion(n_paths: int = 1, n_steps: int = 1000, t_max: float = 1.0, seed: int = 0) -> BrownianMotionResult:
    r"""Sample standard Brownian motion on :math:`[0, t_{\max}]` on a uniform grid.

    Each increment over a step :math:`\Delta t` is an independent
    :math:`\mathcal N(0, \Delta t)` draw, so :math:`W(t) \sim \mathcal N(0, t)`
    exactly at every grid time.

    Parameters
    ----------
    n_paths : int
    n_steps : int
    t_max : float
    seed : int

    Returns
    -------
    BrownianMotionResult

    Examples
    --------
    >>> result = brownian_motion(n_paths=20000, n_steps=100, t_max=2.0, seed=0)
    >>> result.paths.shape
    (20000, 101)
    >>> bool(abs(result.paths[:, -1].var() - 2.0) < 0.1)
    True
    """
    rng = np.random.default_rng(seed)
    dt = t_max / n_steps
    increments = np.sqrt(dt) * rng.standard_normal(size=(n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    np.cumsum(increments, axis=1, out=paths[:, 1:])
    return BrownianMotionResult(times=np.linspace(0.0, t_max, n_steps + 1), paths=paths)


def _unit_steps(rng: np.random.Generator, size, dim: int) -> np.ndarray:
    axis = rng.integers(0, dim, size=size)
    sign = rng.choice(np.array([-1, 1]), size=size)
    return (np.arange(dim) == axis[..., None]) * sign[..., None]


def simple_random_walk(n_walks: int = 1, n_steps: int = 1000, dim: int = 1, seed: int = 0) -> np.ndarray:
    r"""Positions of simple random walks on :math:`\mathbb{Z}^d` started at the origin.

    Parameters
    ----------
    n_walks : int
    n_steps : int
    dim : int
    seed : int

    Returns
    -------
    ndarray of int, shape (n_walks, n_steps + 1, dim)

    Examples
    --------
    >>> walks = simple_random_walk(n_walks=3, n_steps=10, dim=2, seed=0)
    >>> walks.shape
    (3, 11, 2)
    >>> bool(np.all(np.abs(np.diff(walks, axis=1)).sum(axis=2) == 1))
    True
    """
    rng = np.random.default_rng(seed)
    positions = np.zeros((n_walks, n_steps + 1, dim), dtype=np.int64)
    np.cumsum(_unit_steps(rng, (n_walks, n_steps), dim), axis=1, out=positions[:, 1:])
    return positions


def random_walk_return_fraction(dim: int, n_steps: int, n_walks: int = 2000, seed: int = 0) -> float:
    r"""Fraction of simple random walks on :math:`\mathbb{Z}^d` that revisit the origin within ``n_steps`` steps.

    Estimates Pólya's return probability truncated at ``n_steps``. The
    walks are advanced one step at a time, so memory stays
    :math:`O(n_\text{walks}\,d)` however long they run.

    Parameters
    ----------
    dim : int
    n_steps : int
    n_walks : int
    seed : int

    Returns
    -------
    float

    Examples
    --------
    >>> frac = random_walk_return_fraction(dim=1, n_steps=100, n_walks=20000, seed=0)
    >>> bool(abs(frac - return_probability_1d(100)) < 0.01)
    True
    """
    rng = np.random.default_rng(seed)
    position = np.zeros((n_walks, dim), dtype=np.int64)
    returned = np.zeros(n_walks, dtype=bool)
    for _ in range(n_steps):
        position += _unit_steps(rng, n_walks, dim)
        returned |= ~np.any(position, axis=1)
    return float(returned.mean())


def return_probability_1d(n_steps: int) -> float:
    r"""Exact probability that a 1D simple random walk revisits 0 within ``n_steps`` steps.

    A walk avoids 0 for its first :math:`2m` steps with probability
    :math:`\binom{2m}{m}4^{-m} \sim 1/\sqrt{\pi m}`, so the return
    probability is

    .. math::

       P(\text{return by } 2m) = 1 - \binom{2m}{m}4^{-m} \to 1,

    Pólya's recurrence in one dimension (Feller, *An Introduction to
    Probability Theory and Its Applications*, vol. 1, 3rd ed., Sec. III.3).

    Parameters
    ----------
    n_steps : int
        Number of steps; an odd count gives the same answer as ``n_steps - 1``.

    Returns
    -------
    float

    Examples
    --------
    >>> return_probability_1d(2)
    0.5
    >>> round(return_probability_1d(4), 4)
    0.625
    """
    m = n_steps // 2
    log_no_return = special.gammaln(2 * m + 1) - 2 * special.gammaln(m + 1) - 2 * m * np.log(2.0)
    return float(1.0 - np.exp(log_no_return))

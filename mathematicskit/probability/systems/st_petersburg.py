r"""The St. Petersburg paradox and Daniel Bernoulli's expected utility.

A fair coin is tossed until it first lands heads; if that happens on
toss :math:`k`, the game pays :math:`2^k`. The expected payoff
:math:`\sum_k 2^{-k}\,2^k` diverges, yet nobody would pay much to play.
Daniel Bernoulli (1738) resolved this by valuing money through a
concave utility, :math:`u(w) = \ln w`. The payoff simulation uses
:meth:`numpy.random.Generator.geometric`; the certainty equivalent is a
short, rapidly convergent series. See D. Bernoulli, "Specimen theoriae
novae de mensura sortis" (1738).
"""

from __future__ import annotations

import numpy as np

__all__ = ["st_petersburg_payoffs", "st_petersburg_certainty_equivalent"]


def st_petersburg_payoffs(n_games: int, seed: int = 0) -> np.ndarray:
    r"""Simulated payoffs :math:`2^K` of the St. Petersburg game, with :math:`K \sim \mathrm{Geometric}(1/2)`.

    Parameters
    ----------
    n_games : int
        Number of independent games played.
    seed : int

    Returns
    -------
    ndarray, shape (n_games,)
        The payoff of each game, as floats.

    Examples
    --------
    >>> payoffs = st_petersburg_payoffs(5, seed=0)
    >>> bool(payoffs.min() >= 2.0)
    True
    """
    rng = np.random.default_rng(seed)
    k = rng.geometric(0.5, size=n_games)
    return np.exp2(k.astype(np.float64))


def st_petersburg_certainty_equivalent(wealth: float = 0.0, n_terms: int = 200) -> float:
    r"""The sure amount a log-utility player values the St. Petersburg game at.

    With initial wealth :math:`w`, Bernoulli's player is indifferent
    between playing and receiving the sure amount :math:`c` solving

    .. math::

       \ln(w + c) = \sum_{k=1}^{\infty} 2^{-k} \ln(w + 2^k).

    For :math:`w = 0` the right side is :math:`2\ln 2`, so :math:`c = 4`:
    a game with infinite expected value is worth just four ducats. The
    series is truncated after ``n_terms`` terms, far past double
    precision for any reasonable wealth.

    Parameters
    ----------
    wealth : float
        The player's initial wealth :math:`w \geq 0`.
    n_terms : int
        Number of series terms.

    Returns
    -------
    float
        The certainty equivalent :math:`c`.

    Examples
    --------
    >>> round(st_petersburg_certainty_equivalent(0.0), 10)
    4.0
    """
    k = np.arange(1, n_terms + 1, dtype=np.float64)
    expected_utility = float(np.sum(np.exp2(-k) * np.log(wealth + np.exp2(k))))
    return float(np.exp(expected_utility) - wealth)

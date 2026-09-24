r"""Two-player zero-sum matrix games, solved as a pair of linear programs.

John von Neumann's 1928 minimax theorem guarantees that every finite
zero-sum game has a value :math:`v` and optimal mixed strategies
:math:`p, q` with

.. math::

   \max_p \min_q p^T A q = \min_q \max_p p^T A q = v.

Each player's problem is a linear program (the two are LP duals of each
other), so this module solves both with
:func:`~mathematicskit.optimization.systems.linear_programming.linear_program`
rather than hand-rolling a game solver. See J. von Neumann, "Zur Theorie
der Gesellschaftsspiele," Mathematische Annalen 100 (1928), 295-320.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.optimization.core.base import GameResult
from mathematicskit.optimization.systems.linear_programming import linear_program

__all__ = ["solve_zero_sum_game"]


def solve_zero_sum_game(payoff) -> GameResult:
    r"""Optimal mixed strategies and value of a zero-sum matrix game.

    ``payoff[i, j]`` is what the column player pays the row player when
    row ``i`` meets column ``j``. The row player solves

    .. math::

       \max_{p, v} v \quad \text{s.t.} \quad A^T p \geq v \mathbf{1},
       \quad \mathbf{1}^T p = 1, \quad p \geq 0,

    and the column player solves the mirror-image minimization; LP
    duality makes the two optimal values equal, which is von Neumann's
    minimax theorem.

    Parameters
    ----------
    payoff : array_like, shape (m, n)
        Payoff matrix to the row (maximizing) player.

    Returns
    -------
    GameResult

    Examples
    --------
    >>> import numpy as np
    >>> rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # rock-paper-scissors
    >>> result = solve_zero_sum_game(rps)
    >>> np.allclose(result.row_strategy, 1 / 3), abs(round(result.value, 10))
    (True, 0.0)
    """
    a = np.asarray(payoff, dtype=np.float64)
    m, n = a.shape
    # Row player: variables (p_1..p_m, v); minimize -v.
    row = linear_program(
        c=np.r_[np.zeros(m), -1.0],
        a_ub=np.c_[-a.T, np.ones(n)],
        b_ub=np.zeros(n),
        a_eq=np.r_[np.ones(m), 0.0][None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * m + [(None, None)],
    )
    # Column player: variables (q_1..q_n, w); minimize w.
    col = linear_program(
        c=np.r_[np.zeros(n), 1.0],
        a_ub=np.c_[a, -np.ones(m)],
        b_ub=np.zeros(m),
        a_eq=np.r_[np.ones(n), 0.0][None, :],
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * n + [(None, None)],
    )
    return GameResult(row_strategy=row.x[:m], col_strategy=col.x[:n], value=float(row.x[m]))

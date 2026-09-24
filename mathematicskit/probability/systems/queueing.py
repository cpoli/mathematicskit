r"""Erlang's loss formula for telephone traffic.

Calls arrive as a Poisson process and hold a line for an exponential
(in fact any) time; with offered load :math:`A` erlangs (arrival rate
times mean holding time) and :math:`c` lines, a call arriving when all
lines are busy is lost. Erlang (1917) showed that this happens with
probability

.. math::

   B(c, A) = \frac{A^c / c!}{\sum_{i=0}^{c} A^i / i!},

the stationary probability that the :math:`M/M/c/c` birth-death chain
sits in state :math:`c`. It is evaluated with the stable recursion
:math:`B(0) = 1`, :math:`B(i) = A B(i-1) / (i + A B(i-1))`, which never
forms the factorials. See Kleinrock, *Queueing Systems*, vol. 1
(1975), Sec. 3.6.
"""

from __future__ import annotations

__all__ = ["erlang_b"]


def erlang_b(offered_load: float, servers: int) -> float:
    r"""Erlang B blocking probability for ``servers`` lines under ``offered_load`` erlangs.

    Parameters
    ----------
    offered_load : float
        Offered traffic :math:`A = \lambda / \mu \geq 0`, in erlangs.
    servers : int
        Number of lines :math:`c \geq 0`.

    Returns
    -------
    float
        The probability that an arriving call is blocked.

    Examples
    --------
    >>> round(erlang_b(1.0, 1), 4)
    0.5
    >>> round(erlang_b(2.0, 2), 4)
    0.4
    """
    if servers < 0 or offered_load < 0:
        raise ValueError("offered_load and servers must be non-negative.")
    b = 1.0
    for i in range(1, servers + 1):
        b = offered_load * b / (i + offered_load * b)
    return float(b)

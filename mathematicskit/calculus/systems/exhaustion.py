r"""Archimedes' method of exhaustion: bounding :math:`\pi` between the
perimeters of inscribed and circumscribed regular polygons.

Hand-rolled: the doubling recurrence is the historical algorithm itself.
Starting from hexagons around a circle of diameter 1, each step doubles
the number of sides using the harmonic and geometric means,

.. math::

   P_{2n} = \frac{2 p_n P_n}{p_n + P_n}, \qquad p_{2n} = \sqrt{p_n P_{2n}},

where :math:`P_n` and :math:`p_n` are the circumscribed and inscribed
perimeters. See T. L. Heath, *The Works of Archimedes* (Cambridge:
Cambridge University Press, 1897), "Measurement of a Circle."
"""

from __future__ import annotations

import math

from mathematicskit.calculus.core.base import ExhaustionResult

__all__ = ["archimedes_pi_bounds"]


def archimedes_pi_bounds(n_doublings: int = 4) -> ExhaustionResult:
    r"""Lower and upper bounds on :math:`\pi` from regular polygons with :math:`6 \cdot 2^k` sides.

    Four doublings reach Archimedes' 96-gon, which gives
    :math:`3\tfrac{10}{71} < \pi < 3\tfrac{1}{7}`.

    Parameters
    ----------
    n_doublings : int
        Number of side doublings after the initial hexagon.

    Returns
    -------
    ExhaustionResult

    Examples
    --------
    >>> result = archimedes_pi_bounds(4)
    >>> result.sides[-1]
    96
    >>> 3 + 10 / 71 < result.lower[-1] < math.pi < result.upper[-1] < 3 + 1 / 7
    True
    """
    if n_doublings < 0:
        raise ValueError("n_doublings must be >= 0")
    outer, inner = 2 * math.sqrt(3), 3.0
    sides, lower, upper = [6], [inner], [outer]
    for _ in range(n_doublings):
        outer = 2 * inner * outer / (inner + outer)
        inner = math.sqrt(inner * outer)
        sides.append(2 * sides[-1])
        lower.append(inner)
        upper.append(outer)
    return ExhaustionResult(sides=sides, lower=lower, upper=upper)

r"""Group actions on finite sets: orbits and Burnside's orbit-counting lemma.

No numpy/scipy equivalent. See Dummit & Foote, *Abstract Algebra*, 3rd
ed., Sec. 4.1 (group actions), and W. Burnside, *Theory of Groups of
Finite Order* (Cambridge: Cambridge University Press, 1897), Sec. 145.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

from mathematicskit.abstract_algebra.core.base import FiniteGroup

__all__ = ["orbits", "count_orbits"]


def orbits(group: FiniteGroup, points, action: Callable) -> list:
    r"""The orbits of `group` acting on `points`, found by direct search.

    Parameters
    ----------
    group : FiniteGroup
    points : iterable
        A finite set closed under the action.
    action : callable
        ``action(g, x)`` returns the image of point ``x`` under group element ``g``.

    Returns
    -------
    list of list
        Each orbit as a list of points.

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> rotate = lambda g, word: word[g:] + word[:g]  # Z_3 rotating 3-letter words
    >>> len(orbits(CyclicGroup(3), ["aab", "aba", "baa", "aaa"], rotate))
    2
    """
    seen = set()
    result = []
    for x in points:
        if x in seen:
            continue
        orbit = []
        for g in group.elements:
            y = action(g, x)
            if y not in seen:
                seen.add(y)
                orbit.append(y)
        result.append(orbit)
    return result


def count_orbits(group: FiniteGroup, points, action: Callable) -> int:
    r"""The number of orbits, by Burnside's lemma: the average number of fixed points.

    .. math::

       |X/G| = \frac{1}{|G|} \sum_{g \in G} |\mathrm{Fix}(g)|

    The formula was known to Augustin-Louis Cauchy (1845) and Ferdinand
    Georg Frobenius (1887); William Burnside's 1897 book made it
    standard. Counting fixed points is usually far easier than listing
    orbits. See Dummit & Foote, *Abstract Algebra*, 3rd ed., Sec. 4.1.

    Parameters
    ----------
    group : FiniteGroup
    points : iterable
    action : callable
        ``action(g, x)`` returns the image of point ``x`` under ``g``.

    Returns
    -------
    int

    Examples
    --------
    >>> from itertools import product
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> necklaces = ["".join(w) for w in product("RB", repeat=4)]  # 2-colored 4-bead necklaces
    >>> count_orbits(CyclicGroup(4), necklaces, lambda g, w: w[g:] + w[:g])
    6
    """
    points = list(points)
    total = sum(sum(1 for x in points if action(g, x) == x) for g in group.elements)
    count = Fraction(total, group.order)
    if count.denominator != 1:
        raise ValueError("fixed-point average is not an integer; is `action` a genuine group action?")
    return int(count)

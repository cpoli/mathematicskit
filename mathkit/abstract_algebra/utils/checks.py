r"""A small structural check -- supporting numerics for
mathkit.abstract_algebra's systems/ modules, not a model in its own right.
"""

from __future__ import annotations

from mathkit.abstract_algebra.core.base import FiniteGroup

__all__ = ["is_cyclic"]


def is_cyclic(group: FiniteGroup) -> bool:
    r"""Whether `group` is cyclic: some element generates the entire group.

    Checked by testing whether any single element has order
    :math:`|G|` (via :meth:`~mathkit.abstract_algebra.core.base.FiniteGroup.element_order`).
    See Dummit & Foote, *Abstract Algebra*, 3rd ed., Sec. 2.3.

    Parameters
    ----------
    group : FiniteGroup

    Returns
    -------
    bool

    Examples
    --------
    >>> from mathkit.abstract_algebra.systems.groups import CyclicGroup, PermutationGroup
    >>> is_cyclic(CyclicGroup(6))
    True
    >>> is_cyclic(PermutationGroup(3))  # S_3 is not cyclic (it's non-abelian)
    False
    """
    return any(group.element_order(e) == group.order for e in group.elements)

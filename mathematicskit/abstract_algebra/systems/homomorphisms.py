r"""Group homomorphisms: the homomorphism property, kernel, and image.

No numpy/scipy equivalent. See Dummit & Foote, *Abstract Algebra*, 3rd
ed., Sec. 1.6 and 3.3 (the isomorphism theorems), and E. Noether,
"Abstrakter Aufbau der Idealtheorie in algebraischen Zahl- und
Funktionenkörpern," Mathematische Annalen 96 (1927), 26-61.
"""

from __future__ import annotations

from typing import Callable

from mathematicskit.abstract_algebra.core.base import FiniteGroup, HomomorphismResult

__all__ = ["is_homomorphism", "homomorphism_kernel", "homomorphism_image", "analyze_homomorphism"]


def is_homomorphism(source: FiniteGroup, target: FiniteGroup, phi: Callable) -> bool:
    r"""Whether :math:`\varphi(ab) = \varphi(a)\varphi(b)` for every pair :math:`a, b \in G`.

    Parameters
    ----------
    source : FiniteGroup
    target : FiniteGroup
    phi : callable
        Maps elements of `source` to elements of `target`.

    Returns
    -------
    bool

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> is_homomorphism(CyclicGroup(12), CyclicGroup(4), lambda a: a % 4)
    True
    >>> is_homomorphism(CyclicGroup(12), CyclicGroup(5), lambda a: a % 5)
    False
    """
    return all(phi(source.operate(a, b)) == target.operate(phi(a), phi(b)) for a in source.elements for b in source.elements)


def homomorphism_kernel(source: FiniteGroup, target: FiniteGroup, phi: Callable) -> list:
    r"""The kernel :math:`\ker\varphi = \{g \in G : \varphi(g) = e_H\}`, always a normal subgroup.

    Parameters
    ----------
    source : FiniteGroup
    target : FiniteGroup
    phi : callable

    Returns
    -------
    list

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> homomorphism_kernel(CyclicGroup(12), CyclicGroup(4), lambda a: a % 4)
    [0, 4, 8]
    """
    e = target.identity()
    return [g for g in source.elements if phi(g) == e]


def homomorphism_image(source: FiniteGroup, target: FiniteGroup, phi: Callable) -> list:
    r"""The image :math:`\varphi(G) \le H`, in the target group's element order.

    Parameters
    ----------
    source : FiniteGroup
    target : FiniteGroup
    phi : callable

    Returns
    -------
    list

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> homomorphism_image(CyclicGroup(6), CyclicGroup(6), lambda a: (2 * a) % 6)
    [0, 2, 4]
    """
    values = {phi(g) for g in source.elements}
    return [h for h in target.elements if h in values]


def analyze_homomorphism(source: FiniteGroup, target: FiniteGroup, phi: Callable) -> HomomorphismResult:
    r"""Kernel, image, and homomorphism check together.

    The first isomorphism theorem, stated in its modern abstract form by
    Emmy Noether in 1927, gives :math:`G/\ker\varphi \cong \varphi(G)`,
    so :math:`|G| = |\ker\varphi| \cdot |\varphi(G)|`.

    Parameters
    ----------
    source : FiniteGroup
    target : FiniteGroup
    phi : callable

    Returns
    -------
    HomomorphismResult

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import CyclicGroup
    >>> result = analyze_homomorphism(CyclicGroup(12), CyclicGroup(4), lambda a: a % 4)
    >>> len(result.kernel) * len(result.image)  # = |Z_12|
    12
    """
    return HomomorphismResult(
        kernel=homomorphism_kernel(source, target, phi),
        image=homomorphism_image(source, target, phi),
        is_homomorphism=is_homomorphism(source, target, phi),
    )

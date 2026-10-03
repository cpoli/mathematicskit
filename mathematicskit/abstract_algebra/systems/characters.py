r"""Conjugacy classes and character tables of small finite groups.

A character :math:`\chi(g) = \operatorname{tr}\rho(g)` of a
representation is constant on conjugacy classes, and Frobenius (1896)
showed that the irreducible characters of a finite group are as many
as its classes and orthonormal:
:math:`\frac1{|G|}\sum_g \chi_i(g)\overline{\chi_j(g)} = \delta_{ij}`.

Burnside's algorithm (1911, made practical by Dixon in 1967) computes
them without constructing any representation. The class sums
:math:`\hat C_i = \sum_{g \in C_i} g` multiply as
:math:`\hat C_i \hat C_j = \sum_k a_{ijk} \hat C_k`, where
:math:`a_{ijk}` counts the ways an element of :math:`C_k` factors as
:math:`xy` with :math:`x \in C_i`, :math:`y \in C_j`. Each irreducible
:math:`\chi` gives a common eigenvector
:math:`\omega_k = |C_k|\chi(g_k)/\chi(1)` of all the matrices
:math:`(M_i)_{jk} = a_{ijk}`, with eigenvalue :math:`\omega_i`, and
orthogonality then fixes :math:`\chi(1)`. Hand-rolled on the group
classes of :mod:`mathematicskit.abstract_algebra.systems.groups`, with
the eigenvectors from :func:`numpy.linalg.eig`. See F. G. Frobenius,
"Über Gruppencharaktere," Sitzungsberichte der Königlich Preußischen
Akademie der Wissenschaften zu Berlin (1896), 985-1021; J. D. Dixon,
"High Speed Computation of Group Characters," Numerische Mathematik 10
(1967), 446-450; and James & Liebeck, *Representations and Characters
of Groups*, 2nd ed. (2001), Ch. 13-16.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.abstract_algebra.core.base import CharacterTableResult, FiniteGroup

__all__ = ["conjugacy_classes", "character_table"]


def conjugacy_classes(group: FiniteGroup) -> list:
    r"""The conjugacy classes :math:`\{h g h^{-1} : h \in G\}`, identity class first.

    Classes are ordered by the order of their elements, then by size.

    Parameters
    ----------
    group : FiniteGroup

    Returns
    -------
    list of list

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import PermutationGroup
    >>> [len(c) for c in conjugacy_classes(PermutationGroup(4))]  # cycle types of S_4
    [1, 3, 6, 8, 6]
    """
    unseen = list(group.elements)
    classes = []
    while unseen:
        g = unseen[0]
        cls = []
        for h in group.elements:
            c = group.operate(group.operate(h, g), group.inverse(h))
            if c not in cls:
                cls.append(c)
        classes.append(cls)
        unseen = [x for x in unseen if x not in cls]
    return sorted(classes, key=lambda c: (group.element_order(c[0]), len(c)))


def _clean(z: complex, tol: float = 1e-9) -> complex:
    real = 0.0 if abs(z.real) < tol else (round(z.real) if abs(z.real - round(z.real)) < tol else z.real)
    imag = 0.0 if abs(z.imag) < tol else (round(z.imag) if abs(z.imag - round(z.imag)) < tol else z.imag)
    return complex(real, imag)


def character_table(group: FiniteGroup, seed: int = 0) -> CharacterTableResult:
    r"""The irreducible complex characters of a small finite group, by Burnside's algorithm.

    Forms the class-multiplication matrices :math:`M_i`, takes the
    eigenvectors of one random combination :math:`\sum_i r_i M_i` (its
    eigenvalues are distinct with probability one, so each eigenvector is
    a common eigenvector of every :math:`M_i`), normalizes each to
    :math:`\omega_1 = 1`, and recovers

    .. math::

       \chi(1)^2 = \frac{|G|}{\sum_k |\omega_k|^2/|C_k|}, \qquad
       \chi(g_k) = \frac{\chi(1)\,\omega_k}{|C_k|}.

    Values within ``1e-9`` of an integer (or of zero) are rounded to it.

    Parameters
    ----------
    group : FiniteGroup
    seed : int
        Seeds the random combination.

    Returns
    -------
    CharacterTableResult

    Examples
    --------
    >>> from mathematicskit.abstract_algebra.systems.groups import PermutationGroup
    >>> result = character_table(PermutationGroup(3))
    >>> result.class_sizes, result.degrees
    ([1, 3, 2], [1, 1, 2])
    >>> result.table.real.astype(int)
    array([[ 1,  1,  1],
           [ 1, -1,  1],
           [ 2,  0, -1]])
    """
    classes = conjugacy_classes(group)
    r = len(classes)
    index = {g: k for k, cls in enumerate(classes) for g in cls}
    sizes = np.array([len(c) for c in classes], dtype=np.float64)
    # a[i, j, k] = #{x in C_i : x^{-1} z in C_j} for a fixed z in C_k.
    a = np.zeros((r, r, r))
    for k, cls in enumerate(classes):
        z = cls[0]
        for i, ci in enumerate(classes):
            for x in ci:
                a[i, index[group.operate(group.inverse(x), z)], k] += 1
    weights = np.random.default_rng(seed).uniform(1.0, 2.0, size=r)
    combined = np.tensordot(weights, a, axes=1)
    _eigenvalues, vectors = np.linalg.eig(combined)
    rows = []
    for v in vectors.T:
        omega = v / v[0]
        degree = np.sqrt(group.order / np.sum(np.abs(omega) ** 2 / sizes))
        rows.append([_clean(complex(degree * w / s)) for w, s in zip(omega, sizes, strict=True)])
    rows.sort(key=lambda row: (round(row[0].real), any(abs(x - 1) > 1e-9 for x in row), [(-x.real, -x.imag) for x in row]))
    return CharacterTableResult(classes=classes, table=np.array(rows, dtype=complex))

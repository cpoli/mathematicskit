r"""Simplicial homology: the Smith normal form, Betti numbers, torsion, and the Mayer-Vietoris sequence.

The :math:`k`-th homology group of a complex is
:math:`H_k = \ker \partial_k / \operatorname{im} \partial_{k+1}`. Over a
field it is a vector space, and its dimension, the Betti number
:math:`\beta_k = f_k - \operatorname{rk}\partial_k - \operatorname{rk}\partial_{k+1}`,
needs only matrix ranks. Over the integers it can also have torsion,
which the Smith normal form of :math:`\partial_{k+1}` exposes as its
invariant factors greater than 1 (H. J. S. Smith 1861; H. Poincaré,
"Complément à l'Analysis Situs," 1900). Ranks over :math:`\mathbb{Q}`
come from :func:`numpy.linalg.matrix_rank` and cycle spaces from
:func:`scipy.linalg.null_space`; the Smith normal form is hand-rolled in
exact integer arithmetic, which neither numpy nor scipy offers. See
J. R. Munkres, *Elements of Algebraic Topology* (Menlo Park:
Addison-Wesley, 1984), Secs. 11 and 33.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from mathematicskit.topology.core.base import HomologyResult, MayerVietorisResult, SimplicialComplex, SmithNormalFormResult
from mathematicskit.topology.utils.modular import rank_mod_p

__all__ = ["smith_normal_form", "betti_numbers", "homology", "mayer_vietoris"]


def smith_normal_form(A) -> SmithNormalFormResult:
    r"""The Smith normal form :math:`D = UAV` of an integer matrix.

    Repeatedly moves the smallest nonzero entry to the pivot, clears its
    row and column by integer row and column operations (the Euclidean
    algorithm in matrix form), and adds a row back whenever the pivot does
    not divide the rest, until :math:`D` is diagonal with
    :math:`d_1 \mid d_2 \mid \cdots`. The invariant factors classify the
    finitely generated abelian group :math:`\mathbb{Z}^m / A\mathbb{Z}^n`.
    Arithmetic is in exact Python integers. See H. J. S. Smith, "On
    systems of linear indeterminate equations and congruences,"
    Philosophical Transactions of the Royal Society 151 (1861), 293-326.

    Parameters
    ----------
    A : array_like of int, shape (m, n)

    Returns
    -------
    SmithNormalFormResult

    Examples
    --------
    >>> result = smith_normal_form([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
    >>> result.invariant_factors
    [2, 6, 12]
    >>> bool(np.array_equal(result.U @ np.array([[2, 4, 4], [-6, 6, 12], [10, -4, -16]]) @ result.V, result.D))
    True
    """
    arr = np.atleast_2d(np.asarray(A, dtype=object))
    m, n = arr.shape
    M = [[int(x) for x in row] for row in arr]
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def swap_rows(X, i, j):
        X[i], X[j] = X[j], X[i]

    def swap_cols(X, i, j):
        for row in X:
            row[i], row[j] = row[j], row[i]

    for t in range(min(m, n)):
        while True:
            entries = [(abs(M[i][j]), i, j) for i in range(t, m) for j in range(t, n) if M[i][j]]
            if not entries:
                break
            _, i, j = min(entries)
            swap_rows(M, t, i)
            swap_rows(U, t, i)
            swap_cols(M, t, j)
            swap_cols(V, t, j)
            p = M[t][t]
            clean = True
            for i in range(t + 1, m):
                q = M[i][t] // p
                if q:
                    M[i] = [a - q * b for a, b in zip(M[i], M[t], strict=True)]
                    U[i] = [a - q * b for a, b in zip(U[i], U[t], strict=True)]
                clean &= M[i][t] == 0
            for j in range(t + 1, n):
                q = M[t][j] // p
                if q:
                    for X in (M, V):
                        for row in X:
                            row[j] -= q * row[t]
                clean &= M[t][j] == 0
            if not clean:
                continue
            bad = next((i for i in range(t + 1, m) if any(M[i][j] % p for j in range(t + 1, n))), None)
            if bad is None:
                break
            M[t] = [a + b for a, b in zip(M[t], M[bad], strict=True)]
            U[t] = [a + b for a, b in zip(U[t], U[bad], strict=True)]
        if t < m and t < n and M[t][t] < 0:
            M[t] = [-a for a in M[t]]
            U[t] = [-a for a in U[t]]

    def as_array(X, shape):
        return np.array(X, dtype=np.int64).reshape(shape)

    return SmithNormalFormResult(D=as_array(M, (m, n)), U=as_array(U, (m, m)), V=as_array(V, (n, n)))


def _boundary_rank_gf2(K: SimplicialComplex, k: int) -> int:
    """Rank of the boundary map mod 2, eliminating bit-set columns built straight from the simplices (no dense matrix)."""
    if not 1 <= k <= K.dimension:
        return 0
    pivots: dict[int, int] = {}
    for s in K.simplices(k):
        bits = 0
        for i in range(k + 1):
            bits ^= 1 << K.index(s[:i] + s[i + 1 :])
        while bits:
            low = bits.bit_length() - 1
            if low not in pivots:
                pivots[low] = bits
                break
            bits ^= pivots[low]
    return len(pivots)


def _rank(d: np.ndarray, field: int) -> int:
    if d.size == 0:
        return 0
    return int(np.linalg.matrix_rank(d.astype(float))) if field == 0 else rank_mod_p(d, field)


def betti_numbers(K: SimplicialComplex, field: int = 0) -> tuple[int, ...]:
    r"""The Betti numbers :math:`\beta_k = f_k - \operatorname{rk}\partial_k - \operatorname{rk}\partial_{k+1}`.

    Enrico Betti's connectivity numbers (1871), in Poincaré's homological
    form. Over :math:`\mathbb{Q}` (``field=0``) they count the free part
    of integer homology; over :math:`\mathbb{Z}/p` they also pick up
    :math:`p`-torsion, so the Klein bottle has :math:`\beta_1 = 1` over
    :math:`\mathbb{Q}` but 2 over :math:`\mathbb{Z}/2`. See E. Betti,
    "Sopra gli spazi di un numero qualunque di dimensioni," Annali di
    Matematica Pura ed Applicata 4 (1871), 140-158.

    Parameters
    ----------
    K : SimplicialComplex
    field : int
        0 for the rationals (via :func:`numpy.linalg.matrix_rank`), or a prime ``p``.
        ``field=2`` never builds a dense matrix, and is the fast choice for
        large complexes such as Vietoris-Rips complexes.

    Returns
    -------
    tuple of int
        :math:`(\beta_0, \dots, \beta_{\dim K})`.

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import sphere, torus
    >>> betti_numbers(sphere(2)), betti_numbers(torus())
    ((1, 0, 1), (1, 2, 1))
    """
    if field == 2:
        ranks = [_boundary_rank_gf2(K, k) for k in range(K.dimension + 2)]
    else:
        ranks = [_rank(K.boundary_matrix(k), field) for k in range(K.dimension + 2)]
    return tuple(f - ranks[k] - ranks[k + 1] for k, f in enumerate(K.f_vector))


def homology(K: SimplicialComplex) -> HomologyResult:
    r"""Integer homology :math:`H_k(K; \mathbb{Z}) \cong \mathbb{Z}^{\beta_k} \oplus \mathbb{Z}/t_1 \oplus \cdots` by Smith normal form.

    The rank of each boundary matrix gives the Betti numbers; the
    invariant factors of :math:`\partial_{k+1}` that exceed 1 are the
    torsion coefficients of :math:`H_k`, which Poincaré introduced in
    1900 after Heegaard showed that Betti numbers alone miss them. The
    alternating sum of the Betti numbers equals that of the simplex
    counts (the Euler-Poincaré formula).

    Parameters
    ----------
    K : SimplicialComplex
        Small enough for exact integer elimination (hundreds of simplices).

    Returns
    -------
    HomologyResult

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import klein_bottle, projective_plane
    >>> str(homology(projective_plane()))
    'H_0 = Z, H_1 = Z/2, H_2 = 0'
    >>> homology(klein_bottle(3, 3)).group(1)
    'Z + Z/2'
    """
    factors = [smith_normal_form(K.boundary_matrix(k)).invariant_factors for k in range(K.dimension + 2)]
    ranks = [len(f) for f in factors]
    betti = tuple(f - ranks[k] - ranks[k + 1] for k, f in enumerate(K.f_vector))
    torsion = tuple(tuple(d for d in factors[k + 1] if d > 1) for k in range(K.dimension + 1))
    return HomologyResult(betti_numbers=betti, torsion=torsion)


def _inclusion(sub: SimplicialComplex, K: SimplicialComplex, k: int) -> np.ndarray:
    J = np.zeros((len(K.simplices(k)), len(sub.simplices(k))))
    for j, s in enumerate(sub.simplices(k)):
        J[K.index(s), j] = 1.0
    return J


def _matrix_rank(M: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(M)) if M.size else 0


def mayer_vietoris(A: SimplicialComplex, B: SimplicialComplex) -> MayerVietorisResult:
    r"""The ranks in the Mayer-Vietoris sequence of :math:`A \cup B`, over :math:`\mathbb{Q}`.

    .. math::

        \cdots \to H_k(A \cap B) \xrightarrow{i_k} H_k(A) \oplus H_k(B) \to H_k(A \cup B) \xrightarrow{\delta} H_{k-1}(A \cap B) \to \cdots

    The rank of :math:`i_k` is computed directly: the cycles of
    :math:`A \cap B` (a basis from :func:`scipy.linalg.null_space`)
    pushed into :math:`A` and :math:`B`, counted modulo the boundaries
    there. Exactness then predicts the Betti numbers of the union from
    those of the pieces (:attr:`MayerVietorisResult.predicted_union`),
    which the result also computes directly. See W. Mayer, "Über
    abstrakte Topologie," Monatshefte für Mathematik und Physik 36 (1929),
    1-42; L. Vietoris, "Über die Homologiegruppen der Vereinigung zweier
    Komplexe," ibid. 37 (1930), 159-162.

    Parameters
    ----------
    A, B : SimplicialComplex

    Returns
    -------
    MayerVietorisResult

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import circle
    >>> C = circle(6)
    >>> upper, lower = C.induced_subcomplex([0, 1, 2, 3]), C.induced_subcomplex([3, 4, 5, 0])
    >>> result = mayer_vietoris(upper, lower)
    >>> result.betti_intersection, result.betti_union, result.predicted_union
    ((2,), (1, 1), (1, 1))
    """
    union, inter = A | B, A & B
    top = union.dimension
    pad = lambda t: tuple(t) + (0,) * (top + 1 - len(t))  # noqa: E731
    ranks = []
    for k in range(top + 1):
        if not inter.simplices(k):
            ranks.append(0)
            continue
        d = inter.boundary_matrix(k).astype(float)
        cycles = np.eye(len(inter.simplices(k))) if d.shape[0] == 0 else null_space(d)
        image = np.vstack([_inclusion(inter, A, k) @ cycles, _inclusion(inter, B, k) @ cycles])
        bd_a, bd_b = A.boundary_matrix(k + 1).astype(float), B.boundary_matrix(k + 1).astype(float)
        boundaries = np.zeros((image.shape[0], bd_a.shape[1] + bd_b.shape[1]))
        boundaries[: bd_a.shape[0], : bd_a.shape[1]] = bd_a
        boundaries[bd_a.shape[0] :, bd_a.shape[1] :] = bd_b
        ranks.append(_matrix_rank(np.hstack([boundaries, image])) - _matrix_rank(boundaries))
    return MayerVietorisResult(
        betti_a=pad(betti_numbers(A)),
        betti_b=pad(betti_numbers(B)),
        betti_intersection=betti_numbers(inter),
        betti_union=pad(betti_numbers(union)),
        rank_inclusion=tuple(ranks),
    )

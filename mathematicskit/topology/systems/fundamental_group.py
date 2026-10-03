r"""The fundamental group of a simplicial complex, as a presentation by generators and relations.

Poincaré's fundamental group (1895) of a connected complex depends only
on its 2-skeleton. Contracting a spanning tree of the 1-skeleton leaves
one generator per remaining edge and one relation per triangle (the
edge-path group; see E. H. Spanier, *Algebraic Topology*, New York:
McGraw-Hill, 1966, Sec. 3.6). Tietze transformations then eliminate any
generator that appears exactly once in some relation. The abelianization
of the result is :math:`H_1`, computed from the relation matrix by the
Smith normal form. Hand-rolled: there is no numpy/scipy equivalent.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mathematicskit.topology.core.base import FundamentalGroupPresentation, SimplicialComplex, _group_name
from mathematicskit.topology.systems.homology import smith_normal_form

__all__ = ["fundamental_group", "abelianization"]


def _free_reduce(word: list) -> list:
    out: list = []
    for g, e in word:
        if out and out[-1] == (g, -e):
            out.pop()
        else:
            out.append((g, e))
    while len(out) > 1 and out[0] == (out[-1][0], -out[-1][1]):
        out = out[1:-1]
    return out


def _inverse(word: list) -> list:
    return [(g, -e) for g, e in reversed(word)]


def _tietze(n_generators: int, relations: list) -> tuple[list[int], list]:
    alive = list(range(n_generators))
    rels = [r for r in (_free_reduce(r) for r in relations) if r]
    while True:
        candidates = []
        for ri, r in enumerate(rels):
            counts: dict[int, int] = {}
            for g, _ in r:
                counts[g] = counts.get(g, 0) + 1
            candidates += [(len(r), ri, g) for g, c in counts.items() if c == 1]
        if not candidates:
            break
        _, ri, g = min(candidates)
        r = rels.pop(ri)
        pos = next(i for i, (h, _) in enumerate(r) if h == g)
        e = r[pos][1]
        rest = r[pos + 1 :] + r[:pos]  # g^e * rest = 1, so g^e = rest^-1
        value = _inverse(rest) if e == 1 else rest
        substituted = []
        for other in rels:
            word: list = []
            for h, f in other:
                word += (value if f == 1 else _inverse(value)) if h == g else [(h, f)]
            substituted.append(_free_reduce(word))
        rels = [w for w in substituted if w]
        alive.remove(g)
    return alive, rels


def fundamental_group(K: SimplicialComplex, base: int | None = None, simplify: bool = True) -> FundamentalGroupPresentation:
    r"""A presentation of :math:`\pi_1(K, v_0)` from a spanning tree of the 1-skeleton.

    Every edge outside a breadth-first spanning tree of the base point's
    component is a generator (oriented from its lower to its higher
    vertex); every triangle :math:`[a, b, c]` gives the relation
    :math:`[ab][bc][ac]^{-1} = 1`, with tree edges deleted. With
    ``simplify``, Tietze moves remove generators that a relation
    expresses in terms of the others. See H. Poincaré, "Analysis situs,"
    Journal de l'École Polytechnique (2) 1 (1895), 1-121, Sec. 12.

    Parameters
    ----------
    K : SimplicialComplex
    base : int, optional
        Base vertex; the smallest vertex label if omitted.
    simplify : bool
        Apply Tietze transformations to shorten the presentation.

    Returns
    -------
    FundamentalGroupPresentation

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import projective_plane, sphere, torus
    >>> str(fundamental_group(sphere(2))), str(fundamental_group(projective_plane()))
    ('1', '< a | aa >')
    >>> P = fundamental_group(torus(3, 3))
    >>> len(P.generators), len(P.relations), len(P.relations[0])
    (2, 1, 4)
    """
    v0 = K.vertices[0] if base is None else int(base)
    neighbours: dict[int, list[int]] = {v: [] for v in K.vertices}
    for a, b in K.simplices(1):
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen, tree = {v0}, set()
    queue = deque([v0])
    while queue:
        v = queue.popleft()
        for u in neighbours[v]:
            if u not in seen:
                seen.add(u)
                tree.add((min(u, v), max(u, v)))
                queue.append(u)
    edges = [e for e in K.simplices(1) if e[0] in seen and e not in tree]
    gen = {e: i for i, e in enumerate(edges)}
    relations = []
    for a, b, c in K.simplices(2):
        if a in seen:
            word = [(gen[e], s) for e, s in (((a, b), 1), ((b, c), 1), ((a, c), -1)) if e in gen]
            relations.append(word)
    if not simplify:
        return FundamentalGroupPresentation(generators=edges, relations=[_free_reduce(r) for r in relations if _free_reduce(r)])
    alive, rels = _tietze(len(edges), relations)
    renumber = {g: i for i, g in enumerate(alive)}
    rels = [[(renumber[g], e) for g, e in r] for r in rels]
    rels = [_inverse(r) if sum(e for _, e in r) < 0 else r for r in rels]
    return FundamentalGroupPresentation(generators=[edges[g] for g in alive], relations=rels)


def abelianization(presentation: FundamentalGroupPresentation) -> str:
    r"""The abelianized group :math:`\pi_1^{ab} \cong H_1`, e.g. ``"Z + Z/2"``.

    Abelianizing turns each relation into a row of exponent sums; the
    group is :math:`\mathbb{Z}^n` modulo the row space, read off from
    the Smith normal form of that matrix (Hurewicz: it equals
    :math:`H_1`).

    Parameters
    ----------
    presentation : FundamentalGroupPresentation

    Returns
    -------
    str

    Examples
    --------
    >>> from mathematicskit.topology.systems.complexes import klein_bottle
    >>> abelianization(fundamental_group(klein_bottle(3, 3)))
    'Z + Z/2'
    """
    n = len(presentation.generators)
    R = np.zeros((len(presentation.relations), n), dtype=np.int64)
    for i, r in enumerate(presentation.relations):
        for g, e in r:
            R[i, g] += e
    factors = smith_normal_form(R).invariant_factors if R.size else []
    return _group_name(n - len(factors), [d for d in factors if d > 1])

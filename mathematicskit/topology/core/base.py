r"""The simplicial complex and the result containers shared by mathematicskit.topology.

A finite abstract simplicial complex is a family of finite vertex sets
closed under taking subsets; :class:`SimplicialComplex` stores it as
sorted vertex tuples, graded by dimension, together with the signed
boundary matrices :math:`\partial_k` that every homology computation
starts from. The topological algorithms (homology, fundamental group,
surface classification, persistence, fixed points) don't share a common
interface beyond that, so the rest of this module is the dataclasses
their visualizers and examples consume.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SimplicialComplex",
    "Filtration",
    "SmithNormalFormResult",
    "HomologyResult",
    "PersistenceDiagram",
    "FundamentalGroupPresentation",
    "SurfaceClassification",
    "CriticalPointsResult",
    "MayerVietorisResult",
    "MapperResult",
    "TriangleGrid",
    "FixedPointResult",
]


def _closure(simplices: Iterable) -> set[tuple[int, ...]]:
    faces: set[tuple[int, ...]] = set()
    for s in simplices:
        top = tuple(sorted({int(v) for v in s}))
        if not top or top in faces:
            continue
        for k in range(1, len(top) + 1):
            faces.update(itertools.combinations(top, k))
    return faces


class SimplicialComplex:
    r"""A finite abstract simplicial complex on integer vertices.

    Built from any collection of simplices (vertex lists); every face of
    every simplex is added, so ``SimplicialComplex([[0, 1, 2]])`` is the
    filled triangle with its three edges and three vertices. Simplices
    are stored as sorted tuples, which fixes the orientation used by
    :meth:`boundary_matrix`:

    .. math::

        \partial_k [v_0, \dots, v_k] = \sum_{i=0}^{k} (-1)^i [v_0, \dots, \hat v_i, \dots, v_k].

    Parameters
    ----------
    simplices : iterable of iterable of int
        The simplices (typically the maximal ones); faces are added automatically.
    coordinates : array_like, shape (n_vertices, d), optional
        A position for each vertex label (row ``v`` for vertex ``v``),
        used only for plotting and geometric constructions.

    Examples
    --------
    >>> hollow_triangle = SimplicialComplex([[0, 1], [1, 2], [0, 2]])
    >>> hollow_triangle.f_vector, hollow_triangle.euler_characteristic
    ((3, 3), 0)
    >>> hollow_triangle.boundary_matrix(1)
    array([[-1, -1,  0],
           [ 1,  0, -1],
           [ 0,  1,  1]])
    """

    def __init__(self, simplices: Iterable = (), coordinates=None):
        faces = _closure(simplices)
        dim = max((len(s) for s in faces), default=0) - 1
        self._by_dim: list[list[tuple[int, ...]]] = [sorted(s for s in faces if len(s) == k + 1) for k in range(dim + 1)]
        self._index: list[dict[tuple[int, ...], int]] = [{s: i for i, s in enumerate(level)} for level in self._by_dim]
        self.coordinates = coordinates

    @property
    def coordinates(self) -> np.ndarray | None:
        """ndarray or None: Vertex positions, row ``v`` for vertex ``v`` (assignable from any array-like)."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value) -> None:
        self._coordinates = None if value is None else np.asarray(value, dtype=float)

    # -- size and shape ------------------------------------------------------
    @property
    def dimension(self) -> int:
        """int: Largest simplex dimension (``-1`` for the empty complex)."""
        return len(self._by_dim) - 1

    @property
    def vertices(self) -> list[int]:
        """list of int: The vertex labels, in increasing order."""
        return [s[0] for s in self._by_dim[0]] if self._by_dim else []

    @property
    def f_vector(self) -> tuple[int, ...]:
        r"""tuple of int: :math:`(f_0, f_1, \dots)`, the number of simplices of each dimension."""
        return tuple(len(level) for level in self._by_dim)

    @property
    def euler_characteristic(self) -> int:
        r"""int: :math:`\chi = \sum_k (-1)^k f_k`, the alternating count of simplices."""
        return sum((-1) ** k * f for k, f in enumerate(self.f_vector))

    def simplices(self, k: int | None = None) -> list[tuple[int, ...]]:
        """The ``k``-simplices in lexicographic order, or all simplices by dimension if ``k`` is omitted.

        Parameters
        ----------
        k : int, optional

        Returns
        -------
        list of tuple of int
        """
        if k is None:
            return [s for level in self._by_dim for s in level]
        return list(self._by_dim[k]) if 0 <= k <= self.dimension else []

    def index(self, simplex) -> int:
        """Position of ``simplex`` among the simplices of its dimension (the row/column of the boundary matrices).

        Parameters
        ----------
        simplex : iterable of int

        Returns
        -------
        int
        """
        s = tuple(sorted(int(v) for v in simplex))
        return self._index[len(s) - 1][s]

    @property
    def maximal_simplices(self) -> list[tuple[int, ...]]:
        """list of tuple of int: The simplices that are not a face of any other simplex."""
        result: list[tuple[int, ...]] = []
        for k, level in enumerate(self._by_dim):
            covered: set[tuple[int, ...]] = set()
            if k < self.dimension:
                for s in self._by_dim[k + 1]:
                    covered.update(itertools.combinations(s, k + 1))
            result.extend(s for s in level if s not in covered)
        return result

    # -- algebra -------------------------------------------------------------
    def boundary_matrix(self, k: int) -> np.ndarray:
        r"""The signed boundary matrix :math:`\partial_k: C_k \to C_{k-1}`, shape :math:`(f_{k-1}, f_k)`.

        Column ``j`` holds the boundary of the ``j``-th ``k``-simplex.
        :math:`\partial_0` is the zero map to the trivial group, with no
        rows. Consecutive boundary matrices multiply to zero,
        :math:`\partial_{k-1}\partial_k = 0`.

        Parameters
        ----------
        k : int

        Returns
        -------
        ndarray of int
        """
        rows = len(self._by_dim[k - 1]) if 1 <= k <= self.dimension + 1 else 0
        cols = len(self._by_dim[k]) if 0 <= k <= self.dimension else 0
        d = np.zeros((rows, cols), dtype=np.int64)
        if rows and cols:
            index = self._index[k - 1]
            for j, s in enumerate(self._by_dim[k]):
                for i in range(k + 1):
                    d[index[s[:i] + s[i + 1 :]], j] = (-1) ** i
        return d

    # -- subcomplexes --------------------------------------------------------
    def skeleton(self, k: int) -> SimplicialComplex:
        """The subcomplex of all simplices of dimension at most ``k``.

        Parameters
        ----------
        k : int

        Returns
        -------
        SimplicialComplex
        """
        return SimplicialComplex(self.simplices(min(k, self.dimension)) if k >= 0 else [], self.coordinates)

    def induced_subcomplex(self, vertices) -> SimplicialComplex:
        """The full subcomplex on ``vertices``: every simplex all of whose vertices are in the set.

        Parameters
        ----------
        vertices : iterable of int

        Returns
        -------
        SimplicialComplex
        """
        keep = {int(v) for v in vertices}
        return SimplicialComplex([s for s in self.simplices() if set(s) <= keep], self.coordinates)

    def link(self, vertex: int) -> SimplicialComplex:
        r"""The link of a vertex: the faces :math:`\tau` with :math:`v \notin \tau` and :math:`\tau \cup \{v\}` in the complex.

        Parameters
        ----------
        vertex : int

        Returns
        -------
        SimplicialComplex
        """
        v = int(vertex)
        return SimplicialComplex([tuple(u for u in s if u != v) for s in self.simplices() if v in s and len(s) > 1], self.coordinates)

    def __or__(self, other: SimplicialComplex) -> SimplicialComplex:
        coords = self.coordinates if self.coordinates is not None else other.coordinates
        return SimplicialComplex(self.simplices() + other.simplices(), coords)

    def __and__(self, other: SimplicialComplex) -> SimplicialComplex:
        return SimplicialComplex(set(self.simplices()) & set(other.simplices()), self.coordinates)

    # -- container protocol --------------------------------------------------
    def __contains__(self, simplex) -> bool:
        s = tuple(sorted(int(v) for v in simplex))
        return 0 < len(s) <= len(self._by_dim) and s in self._index[len(s) - 1]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self.simplices())

    def __len__(self) -> int:
        return sum(self.f_vector)

    def __eq__(self, other) -> bool:
        return isinstance(other, SimplicialComplex) and self._by_dim == other._by_dim

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"SimplicialComplex(dimension={self.dimension}, f_vector={self.f_vector})"


@dataclass
class Filtration:
    r"""A nested sequence of simplicial complexes, as simplices with the value at which each appears.

    Every face enters no later than its cofaces, and simplices are kept
    sorted by (value, dimension), so any prefix of :attr:`simplices` is a
    simplicial complex.
    """

    simplices: list
    """list of tuple of int: The simplices in order of appearance."""

    values: np.ndarray
    """ndarray: The filtration value at which each simplex enters."""

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        simplices = [tuple(sorted(int(v) for v in s)) for s in self.simplices]
        order = sorted(range(len(simplices)), key=lambda i: (values[i], len(simplices[i]), simplices[i]))
        self.simplices = [simplices[i] for i in order]
        self.values = values[order]

    def __len__(self) -> int:
        return len(self.simplices)

    def complex_at(self, t: float) -> SimplicialComplex:
        """The subcomplex of simplices with value at most ``t``.

        Parameters
        ----------
        t : float

        Returns
        -------
        SimplicialComplex
        """
        return SimplicialComplex([s for s, v in zip(self.simplices, self.values, strict=True) if v <= t])


@dataclass
class SmithNormalFormResult:
    r"""The Smith normal form :math:`D = UAV` of an integer matrix.

    :math:`U` and :math:`V` are unimodular (integer, determinant
    :math:`\pm 1`) and :math:`D` is diagonal with
    :math:`d_1 \mid d_2 \mid \dots \mid d_r`, all positive.
    """

    D: np.ndarray
    """ndarray: The diagonal form, same shape as the input."""

    U: np.ndarray
    """ndarray: Unimodular row transformation."""

    V: np.ndarray
    """ndarray: Unimodular column transformation."""

    @property
    def invariant_factors(self) -> list[int]:
        r"""list of int: The nonzero diagonal entries :math:`d_1 \mid d_2 \mid \dots`."""
        return [int(d) for d in np.diagonal(self.D) if d != 0]

    @property
    def rank(self) -> int:
        """int: Number of nonzero invariant factors."""
        return len(self.invariant_factors)


def _group_name(rank: int, torsion) -> str:
    parts = ["Z" if rank == 1 else f"Z^{rank}"] if rank else []
    parts += [f"Z/{t}" for t in torsion]
    return " + ".join(parts) if parts else "0"


@dataclass
class HomologyResult:
    r"""Integer homology :math:`H_k \cong \mathbb{Z}^{\beta_k} \oplus \mathbb{Z}/t_1 \oplus \dots` in each dimension."""

    betti_numbers: tuple
    r"""tuple of int: :math:`\beta_k`, the rank of the free part of :math:`H_k`."""

    torsion: tuple
    """tuple of tuple of int: The torsion coefficients of each :math:`H_k` (invariant factors greater than 1)."""

    def group(self, k: int) -> str:
        """Human-readable :math:`H_k`, e.g. ``"Z + Z/2"``.

        Parameters
        ----------
        k : int

        Returns
        -------
        str
        """
        if k >= len(self.betti_numbers):
            return "0"
        return _group_name(self.betti_numbers[k], self.torsion[k])

    @property
    def euler_characteristic(self) -> int:
        r"""int: :math:`\sum_k (-1)^k \beta_k`, equal to the alternating simplex count."""
        return sum((-1) ** k * b for k, b in enumerate(self.betti_numbers))

    def __str__(self) -> str:
        return ", ".join(f"H_{k} = {self.group(k)}" for k in range(len(self.betti_numbers)))


@dataclass
class PersistenceDiagram:
    r"""Birth-death pairs of a filtration's homology classes, per dimension.

    A class that never dies has death :math:`+\infty`.
    """

    pairs: dict
    """dict: ``{k: ndarray of shape (n, 2)}``, the (birth, death) pairs of :math:`H_k`."""

    def diagram(self, k: int) -> np.ndarray:
        """The (birth, death) pairs in dimension ``k``, shape ``(n, 2)``.

        Parameters
        ----------
        k : int

        Returns
        -------
        ndarray
        """
        return self.pairs.get(k, np.empty((0, 2)))

    @property
    def max_dimension(self) -> int:
        """int: Highest homology dimension recorded."""
        return max(self.pairs, default=-1)

    def betti_numbers(self, t: float) -> tuple[int, ...]:
        r"""The Betti numbers of the complex at filtration value ``t``: intervals with birth :math:`\le t <` death.

        Parameters
        ----------
        t : float

        Returns
        -------
        tuple of int
        """
        return tuple(int(np.sum((d[:, 0] <= t) & (t < d[:, 1]))) for d in (self.diagram(k) for k in range(self.max_dimension + 1)))

    def persistence(self, k: int) -> np.ndarray:
        """Lifetimes ``death - birth`` in dimension ``k``.

        Parameters
        ----------
        k : int

        Returns
        -------
        ndarray
        """
        d = self.diagram(k)
        return d[:, 1] - d[:, 0]


@dataclass
class FundamentalGroupPresentation:
    r"""A presentation :math:`\langle g_1, \dots, g_n \mid r_1, \dots, r_m \rangle` of :math:`\pi_1`.

    Each relation is a word, a list of ``(generator, exponent)`` letters
    with exponent :math:`\pm 1`. ``str()`` writes generators as
    ``a, b, c, ...`` and inverses as capitals.
    """

    generators: list
    """list: The edge (or derived label) behind each generator."""

    relations: list = field(default_factory=list)
    """list of list of (int, int): The relators, each equal to the identity."""

    @staticmethod
    def _letter(g: int) -> str:
        return chr(ord("a") + g) if g < 26 else f"g{g}"

    def word(self, relation) -> str:
        """Write a relation as a string, inverses in capitals (``g12`` / ``G12`` past ``z``).

        Parameters
        ----------
        relation : list of (int, int)

        Returns
        -------
        str
        """
        return "".join(self._letter(g) if e > 0 else self._letter(g).upper() for g, e in relation)

    def __str__(self) -> str:
        gens = ", ".join(self._letter(g) for g in range(len(self.generators)))
        rels = ", ".join(self.word(r) for r in self.relations)
        if not gens:
            return "1"
        return f"< {gens} | {rels} >" if rels else f"< {gens} >"


@dataclass
class SurfaceClassification:
    """Where a compact connected surface sits in the classification theorem."""

    name: str
    """str: E.g. ``"torus"``, ``"Klein bottle"``, ``"Möbius strip"``."""

    orientable: bool
    """bool: Whether a consistent orientation of the triangles exists."""

    euler_characteristic: int
    r"""int: :math:`\chi = V - E + F`."""

    boundary_components: int
    """int: Number of boundary circles."""

    genus: int
    """int: Number of handles (orientable) or of cross-caps (non-orientable)."""


@dataclass
class CriticalPointsResult:
    r"""Banchoff's critical points of a height function on the vertices of a complex.

    The index of a vertex is the Euler characteristic of its lower star
    (the simplices whose highest vertex it is), :math:`1 - \chi(\text{lower link})`.
    """

    index: np.ndarray
    """ndarray of int: The index of each vertex (0 for regular points)."""

    minima: list
    """list of int: Vertices whose lower link is empty."""

    saddles: list
    """list of int: Vertices of negative index."""

    maxima: list
    """list of int: Vertices whose lower link is their whole link, a sphere."""

    @property
    def euler_characteristic(self) -> int:
        """int: The total index, equal to the Euler characteristic of the complex."""
        return int(np.sum(self.index))


@dataclass
class MayerVietorisResult:
    r"""The Betti numbers in the Mayer-Vietoris sequence of :math:`X = A \cup B` over :math:`\mathbb{Q}`.

    Exactness at each term gives
    :math:`\beta_k(A \cup B) = \beta_k(A) + \beta_k(B) - \operatorname{rk} i_k + \beta_{k-1}(A \cap B) - \operatorname{rk} i_{k-1}`,
    where :math:`i_k: H_k(A \cap B) \to H_k(A) \oplus H_k(B)` is induced by the inclusions.
    """

    betti_a: tuple
    betti_b: tuple
    betti_intersection: tuple
    betti_union: tuple
    rank_inclusion: tuple
    r"""tuple of int: :math:`\operatorname{rk} i_k` for each :math:`k`."""

    @property
    def predicted_union(self) -> tuple[int, ...]:
        """tuple of int: The Betti numbers of the union predicted by exactness of the sequence."""
        n = len(self.betti_union)
        get = lambda t, k: t[k] if 0 <= k < len(t) else 0  # noqa: E731
        return tuple(
            get(self.betti_a, k) + get(self.betti_b, k) - get(self.rank_inclusion, k) + get(self.betti_intersection, k - 1) - get(self.rank_inclusion, k - 1)
            for k in range(n)
        )


@dataclass
class MapperResult:
    """The Mapper graph of a point cloud: one node per cluster of each cover set, an edge where clusters share points."""

    nodes: list
    """list of ndarray: The point indices in each node's cluster."""

    edges: list
    """list of (int, int): Pairs of nodes whose clusters overlap."""

    node_values: np.ndarray
    """ndarray: Mean lens value of each node."""

    @property
    def graph(self) -> SimplicialComplex:
        """SimplicialComplex: The Mapper graph as a 1-dimensional complex."""
        return SimplicialComplex([[i] for i in range(len(self.nodes))] + [list(e) for e in self.edges])


@dataclass
class TriangleGrid:
    r"""The subdivision of the standard triangle into :math:`n^2` small triangles.

    Vertices are integer barycentric coordinates :math:`(i, j, k)` with
    :math:`i + j + k = n`.
    """

    n: int
    """int: Number of subdivisions per side."""

    points: np.ndarray
    """ndarray of int, shape (N, 3): Barycentric coordinates times ``n``."""

    triangles: np.ndarray
    """ndarray of int, shape (n**2, 3): Vertex indices of each small triangle."""

    @property
    def cartesian(self) -> np.ndarray:
        r"""ndarray, shape (N, 2): Vertex positions in the plane, corners at :math:`(0, 0), (1, 0), (\tfrac12, \tfrac{\sqrt3}{2})`."""
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        return self.points @ corners / self.n


@dataclass
class FixedPointResult:
    """An approximate fixed point from a fully labelled triangle of a Sperner labelling."""

    point: np.ndarray
    """ndarray, shape (3,): Barycentric coordinates of the approximate fixed point."""

    triangle: np.ndarray
    """ndarray, shape (3, 3): Barycentric corners of the fully labelled triangle containing it."""

    n: int
    """int: Subdivisions per side; the triangle has side ``1/n``."""

    residual: float
    r"""float: :math:`\max_i |f(x)_i - x_i|` at :attr:`point`."""

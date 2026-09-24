"""Tests for spanning-tree counts, Hamiltonian cycles, bipartite matching,
Turán graphs, assignment, components, A* search, and PageRank."""

from itertools import permutations

import numpy as np
import pytest

from mathematicskit.graph_theory.core.base import Graph
from mathematicskit.graph_theory.systems.assignment import solve_assignment
from mathematicskit.graph_theory.systems.components import connected_components, giant_component_fraction
from mathematicskit.graph_theory.systems.enumeration import count_spanning_trees
from mathematicskit.graph_theory.systems.extremal import clique_number, turan_graph, turan_number
from mathematicskit.graph_theory.systems.hamiltonian import hamiltonian_cycle
from mathematicskit.graph_theory.systems.matching import bipartite_matching
from mathematicskit.graph_theory.systems.ranking import pagerank
from mathematicskit.graph_theory.systems.search import astar_shortest_path
from mathematicskit.graph_theory.systems.shortest_paths import dijkstra_shortest_paths, floyd_warshall_shortest_paths
from mathematicskit.graph_theory.utils.generators import (
    complete_bipartite_graph,
    complete_graph,
    cycle_graph,
    dodecahedron_graph,
    grid_graph,
    path_graph,
    random_graph,
)


@pytest.mark.parametrize("n", range(2, 9))
def test_matrix_tree_theorem_reproduces_cayley(n):
    assert count_spanning_trees(complete_graph(n)) == n ** (n - 2)


def test_spanning_trees_of_cycle_path_and_bipartite():
    assert count_spanning_trees(cycle_graph(7)) == 7
    assert count_spanning_trees(path_graph(6)) == 1
    assert count_spanning_trees(complete_bipartite_graph(3, 4)) == 3 ** (4 - 1) * 4 ** (3 - 1)
    disconnected = Graph(4)
    disconnected.add_edge(0, 1)
    assert count_spanning_trees(disconnected) == 0


def _is_hamiltonian_cycle(graph, cycle):
    return sorted(cycle) == list(range(graph.n_vertices)) and all(cycle[(k + 1) % len(cycle)] in graph.neighbors(cycle[k]) for k in range(len(cycle)))


def test_icosian_game_has_a_solution():
    g = dodecahedron_graph()
    assert all(len(g.neighbors(v)) == 3 for v in range(20))
    assert _is_hamiltonian_cycle(g, hamiltonian_cycle(g))


def test_petersen_graph_is_not_hamiltonian():
    g = Graph(10)
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
        g.add_edge(i, i + 5)
        g.add_edge(5 + i, 5 + (i + 2) % 5)
    assert hamiltonian_cycle(g) is None


def test_hamiltonian_cycle_in_complete_graph():
    g = complete_graph(7)
    assert _is_hamiltonian_cycle(g, hamiltonian_cycle(g, start=3))


def test_konig_matching_equals_vertex_cover_on_random_bipartite_graphs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = Graph(12)
        for u in range(6):
            for v in range(6, 12):
                if rng.random() < 0.3:
                    g.add_edge(u, v)
        result = bipartite_matching(g, left=range(6))
        assert result.size == len(result.vertex_cover)
        cover = set(result.vertex_cover)
        assert all(u in cover or v in cover for u, v, _ in g.edges())
        assert len(set(result.pairs.values())) == result.size


def test_bipartite_matching_rejects_non_bipartite_edge():
    g = Graph(3)
    g.add_edge(0, 1)
    with pytest.raises(ValueError):
        bipartite_matching(g, left=[0, 1])


@pytest.mark.parametrize("n,r", [(6, 2), (7, 3), (10, 4), (9, 3)])
def test_turan_graph_edge_count_and_clique_number(n, r):
    g = turan_graph(n, r)
    assert len(g.edges()) == turan_number(n, r)
    assert clique_number(g) == r


def test_adding_any_edge_to_turan_graph_creates_a_larger_clique():
    g = turan_graph(7, 2)
    missing = [(u, v) for u in range(7) for v in range(u + 1, 7) if v not in g.neighbors(u)]
    for u, v in missing:
        h = turan_graph(7, 2)
        h.add_edge(u, v)
        assert clique_number(h) == 3


def test_assignment_matches_brute_force():
    rng = np.random.default_rng(1)
    cost = rng.integers(0, 20, size=(6, 6))
    best = min(sum(cost[i, p[i]] for i in range(6)) for p in permutations(range(6)))
    assert solve_assignment(cost).total_cost == best
    assert solve_assignment(cost, maximize=True).total_cost == max(sum(cost[i, p[i]] for i in range(6)) for p in permutations(range(6)))


def test_components_and_giant_component_threshold():
    two = Graph(5)
    two.add_edge(0, 1)
    two.add_edge(2, 3)
    result = connected_components(two)
    assert result.n_components == 3
    assert result.sizes.tolist() == [2, 2, 1]
    n = 2000
    below = giant_component_fraction(random_graph(n, 0.5 / n, seed=0))
    above = giant_component_fraction(random_graph(n, 2.0 / n, seed=0))
    assert below < 0.05
    assert above == pytest.approx(0.797, abs=0.05)  # root of s = 1 - exp(-2 s)


def test_astar_matches_dijkstra_and_expands_fewer_vertices():
    blocked = [(r, 10) for r in range(0, 15)]
    g, pos = grid_graph(20, 20, blocked=blocked)
    source, target = 0, 20 * 20 - 1
    manhattan = lambda v: float(np.abs(pos[v] - pos[target]).sum())  # noqa: E731
    guided = astar_shortest_path(g, source, target, manhattan)
    blind = astar_shortest_path(g, source, target)
    assert guided.distance == blind.distance == dijkstra_shortest_paths(g, sources=source).distances[target]
    assert guided.n_expanded < blind.n_expanded
    assert guided.path[0] == source and guided.path[-1] == target


def test_astar_unreachable_target():
    g = Graph(3)
    g.add_edge(0, 1)
    result = astar_shortest_path(g, 0, 2)
    assert result.path == [] and result.distance == float("inf")


def test_floyd_warshall_matches_repeated_dijkstra():
    g = random_graph(15, 0.3, seed=2)
    all_pairs = floyd_warshall_shortest_paths(g).distances
    assert np.allclose(all_pairs, dijkstra_shortest_paths(g).distances)


def test_pagerank_matches_dominant_eigenvector():
    rng = np.random.default_rng(3)
    g = Graph(8, directed=True)
    for u in range(8):
        for v in range(8):
            if u != v and rng.random() < 0.3:
                g.add_edge(u, v)
    d = 0.85
    result = pagerank(g, damping=d)
    a = g.to_sparse().toarray()
    out = a.sum(axis=1)
    p = np.where(out[:, None] > 0, a / np.where(out[:, None] > 0, out[:, None], 1), 1 / 8)
    google = d * p + (1 - d) / 8
    values, vectors = np.linalg.eig(google.T)
    stationary = np.real(vectors[:, np.argmax(np.real(values))])
    stationary /= stationary.sum()
    assert result.scores.sum() == pytest.approx(1.0)
    assert np.allclose(result.scores, stationary, atol=1e-10)


def test_pagerank_uniform_on_symmetric_cycle():
    assert np.allclose(pagerank(cycle_graph(6)).scores, 1 / 6)

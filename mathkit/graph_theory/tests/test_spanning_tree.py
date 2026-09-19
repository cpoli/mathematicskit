"""Tests for Kruskal's and Prim's MST algorithms against a hand-verified
minimum spanning tree."""

from mathkit.graph_theory.core.base import Graph
from mathkit.graph_theory.systems.spanning_tree import kruskal_mst, prim_mst


def _sample_graph():
    g = Graph(5)
    g.add_edge(0, 1, 2.0)
    g.add_edge(0, 3, 6.0)
    g.add_edge(1, 2, 3.0)
    g.add_edge(1, 3, 8.0)
    g.add_edge(1, 4, 5.0)
    g.add_edge(2, 4, 7.0)
    g.add_edge(3, 4, 9.0)
    return g


def test_kruskal_matches_hand_verified_total_weight():
    g = _sample_graph()
    result = kruskal_mst(g)
    # MST: 0-1(2), 1-2(3), 1-4(5), 0-3(6) = 16
    assert result.total_weight == 16.0
    assert len(result.edges) == 4


def test_prim_matches_kruskal_total_weight():
    g = _sample_graph()
    kruskal_result = kruskal_mst(g)
    prim_result = prim_mst(g)
    assert prim_result.total_weight == kruskal_result.total_weight
    assert len(prim_result.edges) == len(kruskal_result.edges)


def test_prim_from_different_start_gives_same_total_weight():
    g = _sample_graph()
    for start in range(5):
        result = prim_mst(g, start=start)
        assert result.total_weight == 16.0


def test_mst_edge_count_is_n_minus_one():
    g = _sample_graph()
    assert len(kruskal_mst(g).edges) == g.n_vertices - 1
    assert len(prim_mst(g).edges) == g.n_vertices - 1

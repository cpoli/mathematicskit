"""Tests for shortest-path algorithms against hand-verified/closed-form
distances."""

import numpy as np

from mathkit.graph_theory.core.base import Graph
from mathkit.graph_theory.systems.shortest_paths import bellman_ford_shortest_paths, dijkstra_shortest_paths, floyd_warshall_shortest_paths


def _sample_graph():
    g = Graph(5)
    g.add_edge(0, 1, 4.0)
    g.add_edge(0, 2, 1.0)
    g.add_edge(2, 1, 2.0)
    g.add_edge(1, 3, 1.0)
    g.add_edge(2, 3, 5.0)
    g.add_edge(3, 4, 3.0)
    return g


def test_dijkstra_matches_hand_computed_distances():
    g = _sample_graph()
    result = dijkstra_shortest_paths(g, sources=0)
    # shortest path 0->1 is via 2: 1+2=3, not direct edge weight 4.
    np.testing.assert_allclose(result.distances, [0.0, 3.0, 1.0, 4.0, 7.0])


def test_dijkstra_and_bellman_ford_agree_on_nonnegative_weights():
    g = _sample_graph()
    dijkstra_result = dijkstra_shortest_paths(g, sources=0)
    bf_result = bellman_ford_shortest_paths(g, sources=0)
    np.testing.assert_allclose(dijkstra_result.distances, bf_result.distances)


def test_bellman_ford_handles_negative_weights():
    g = Graph(3, directed=True)
    g.add_edge(0, 1, 4.0)
    g.add_edge(0, 2, 5.0)
    g.add_edge(1, 2, -2.0)
    result = bellman_ford_shortest_paths(g, sources=0)
    np.testing.assert_allclose(result.distances, [0.0, 4.0, 2.0])


def test_floyd_warshall_matches_dijkstra_all_pairs():
    g = _sample_graph()
    fw_result = floyd_warshall_shortest_paths(g)
    for source in range(5):
        dijkstra_result = dijkstra_shortest_paths(g, sources=source)
        np.testing.assert_allclose(fw_result.distances[source], dijkstra_result.distances)


def test_unreachable_vertex_has_infinite_distance():
    g = Graph(3, directed=True)
    g.add_edge(0, 1, 1.0)
    result = dijkstra_shortest_paths(g, sources=0)
    assert np.isinf(result.distances[2])

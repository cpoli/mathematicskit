"""Tests for maximum flow / minimum cut against hand-verified/known results."""

import numpy as np

from mathematicskit.graph_theory.core.base import Graph
from mathematicskit.graph_theory.systems.max_flow import max_flow_min_cut


def _classic_flow_network():
    # The classic CLRS-style flow network.
    g = Graph(6, directed=True)
    g.add_edge(0, 1, 16)
    g.add_edge(0, 2, 13)
    g.add_edge(1, 2, 10)
    g.add_edge(2, 1, 4)
    g.add_edge(1, 3, 12)
    g.add_edge(2, 4, 14)
    g.add_edge(3, 2, 9)
    g.add_edge(4, 3, 7)
    g.add_edge(3, 5, 20)
    g.add_edge(4, 5, 4)
    return g


def test_max_flow_matches_known_value():
    g = _classic_flow_network()
    result = max_flow_min_cut(g, source=0, sink=5)
    assert result.flow_value == 23.0


def test_min_cut_capacity_equals_max_flow_value():
    g = _classic_flow_network()
    result = max_flow_min_cut(g, source=0, sink=5)
    source_side, sink_side = result.min_cut
    capacity = g.to_sparse().toarray()
    cut_capacity = sum(capacity[u, v] for u in source_side for v in sink_side)
    assert cut_capacity == result.flow_value


def test_min_cut_partitions_include_source_and_sink():
    g = _classic_flow_network()
    result = max_flow_min_cut(g, source=0, sink=5)
    source_side, sink_side = result.min_cut
    assert 0 in source_side
    assert 5 in sink_side
    assert set(source_side) | set(sink_side) == set(range(6))
    assert not (set(source_side) & set(sink_side))


def test_flow_conservation_at_intermediate_vertices():
    g = _classic_flow_network()
    result = max_flow_min_cut(g, source=0, sink=5)
    flow = result.flow_matrix
    for v in [1, 2, 3, 4]:
        inflow = np.sum(flow[:, v])
        outflow = np.sum(flow[v, :])
        assert inflow == outflow


def test_disconnected_source_sink_has_zero_flow():
    g = Graph(3, directed=True)
    g.add_edge(0, 1, 5)
    result = max_flow_min_cut(g, source=0, sink=2)
    assert result.flow_value == 0.0

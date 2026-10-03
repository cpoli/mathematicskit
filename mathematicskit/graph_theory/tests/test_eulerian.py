"""Tests for Hierholzer's Eulerian circuits and trails."""

from collections import Counter

from mathematicskit.graph_theory.core.base import Graph
from mathematicskit.graph_theory.systems.eulerian import eulerian_circuit, eulerian_trail, odd_degree_vertices
from mathematicskit.graph_theory.utils.generators import complete_graph, cycle_graph, grid_graph, path_graph


def _uses_every_edge_once(graph, walk):
    steps = [(u, v) if graph.directed else frozenset((u, v)) for u, v in zip(walk, walk[1:], strict=False)]
    edges = [(u, v) if graph.directed else frozenset((u, v)) for u, v, _ in graph.edges()]
    return Counter(steps) == Counter(edges)


def test_complete_graphs_on_odd_vertex_counts_are_eulerian():
    for n in (3, 5, 7, 9):
        g = complete_graph(n)
        circuit = eulerian_circuit(g)
        assert circuit is not None and circuit[0] == circuit[-1]
        assert _uses_every_edge_once(g, circuit)


def test_odd_vertices_block_circuits():
    assert eulerian_circuit(complete_graph(4)) is None
    assert odd_degree_vertices(complete_graph(4)) == [0, 1, 2, 3]
    assert eulerian_trail(complete_graph(4)) is None


def test_open_trail_runs_between_the_two_odd_vertices():
    g = path_graph(6)
    trail = eulerian_trail(g)
    assert trail == [0, 1, 2, 3, 4, 5]
    assert eulerian_trail(g, start=5) == [5, 4, 3, 2, 1, 0]
    assert eulerian_trail(g, start=2) is None
    assert eulerian_circuit(g) is None


def test_two_triangles_sharing_a_vertex_need_splicing():
    g = Graph(5)
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)]:
        g.add_edge(u, v)
    circuit = eulerian_circuit(g, start=0)
    assert circuit[0] == circuit[-1] == 0
    assert _uses_every_edge_once(g, circuit)


def test_disconnected_edges_have_no_circuit():
    g = Graph(6)
    for u, v in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        g.add_edge(u, v)
    assert eulerian_circuit(g) is None


def test_isolated_vertices_are_ignored():
    g = Graph(5)
    for u, v in [(1, 2), (2, 3), (3, 1)]:
        g.add_edge(u, v)
    circuit = eulerian_circuit(g)
    assert circuit[0] == 1 and len(circuit) == 4


def test_grid_and_cycle():
    g, _ = grid_graph(3, 3)
    assert len(odd_degree_vertices(g)) == 4 and eulerian_trail(g) is None
    circuit = eulerian_circuit(cycle_graph(8))
    assert len(circuit) == 9 and _uses_every_edge_once(cycle_graph(8), circuit)


def test_directed_de_bruijn_circuit():
    # The de Bruijn graph B(2, 3): vertices are 3-bit words, edges append a bit; every in-degree equals out-degree.
    g = Graph(8, directed=True)
    for w in range(8):
        for bit in (0, 1):
            g.add_edge(w, ((w << 1) | bit) & 7)
    circuit = eulerian_circuit(g)
    assert len(circuit) == 17 and _uses_every_edge_once(g, circuit)
    sequence = "".join(str(w & 1) for w in circuit[1:])
    words = {(sequence + sequence[:3])[i : i + 4] for i in range(16)}
    assert len(words) == 16  # a de Bruijn sequence: every 4-bit word appears once


def test_directed_open_trail_and_imbalance():
    g = Graph(3, directed=True)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert eulerian_trail(g) == [0, 1, 2]
    assert eulerian_circuit(g) is None
    g.add_edge(0, 2)
    assert eulerian_trail(g) is None  # vertex 0 has two extra out-edges


def test_edgeless_graph():
    assert eulerian_circuit(Graph(3)) == [0]

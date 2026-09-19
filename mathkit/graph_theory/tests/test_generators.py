"""Tests for graph generators against closed-form edge counts."""

from mathkit.graph_theory.utils.generators import complete_graph, cycle_graph, path_graph, random_graph


def test_complete_graph_edge_count():
    for n in (3, 5, 10):
        assert len(complete_graph(n).edges()) == n * (n - 1) // 2


def test_cycle_graph_edge_count():
    for n in (3, 5, 10):
        assert len(cycle_graph(n).edges()) == n


def test_path_graph_edge_count():
    for n in (3, 5, 10):
        assert len(path_graph(n).edges()) == n - 1


def test_random_graph_reproducibility():
    g1 = random_graph(10, p=0.4, seed=0)
    g2 = random_graph(10, p=0.4, seed=0)
    assert g1.edges() == g2.edges()


def test_random_graph_p_zero_has_no_edges():
    g = random_graph(10, p=0.0, seed=0)
    assert len(g.edges()) == 0


def test_random_graph_p_one_is_complete():
    g = random_graph(6, p=1.0, seed=0)
    assert len(g.edges()) == 6 * 5 // 2

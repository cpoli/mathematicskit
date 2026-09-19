"""Tests for the Graph container against known/hand-verified adjacency structure."""

import numpy as np
import pytest

from mathematicskit.graph_theory.core.base import Graph


def test_undirected_edge_is_symmetric():
    g = Graph(3)
    g.add_edge(0, 1, weight=2.0)
    assert g.neighbors(0)[1] == 2.0
    assert g.neighbors(1)[0] == 2.0


def test_directed_edge_is_one_way():
    g = Graph(3, directed=True)
    g.add_edge(0, 1, weight=2.0)
    assert 1 in g.neighbors(0)
    assert 0 not in g.neighbors(1)


def test_edges_list_undirected_has_no_duplicates():
    g = Graph(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    assert len(g.edges()) == 2


def test_to_sparse_matches_hand_built_matrix():
    g = Graph(3)
    g.add_edge(0, 1, 2.0)
    g.add_edge(1, 2, 3.0)
    expected = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
    np.testing.assert_array_equal(g.to_sparse().toarray(), expected)


def test_default_weight_is_one():
    g = Graph(2)
    g.add_edge(0, 1)
    assert g.neighbors(0)[1] == 1.0


def test_updating_existing_edge_overwrites_weight():
    g = Graph(2)
    g.add_edge(0, 1, 5.0)
    g.add_edge(0, 1, 9.0)
    assert g.neighbors(0)[1] == 9.0


@pytest.mark.parametrize("n", [1, 5, 10])
def test_n_vertices_recorded(n):
    g = Graph(n)
    assert g.n_vertices == n

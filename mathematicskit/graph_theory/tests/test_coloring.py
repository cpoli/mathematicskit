"""Tests for graph coloring against known chromatic numbers."""

from mathematicskit.graph_theory.core.base import Graph
from mathematicskit.graph_theory.systems.coloring import backtracking_coloring, greedy_coloring
from mathematicskit.graph_theory.utils.generators import complete_graph, cycle_graph


def _is_proper_coloring(graph, coloring):
    return all(coloring[u] != coloring[v] for u, v, _w in graph.edges())


def test_complete_graph_needs_n_colors():
    g = complete_graph(5)
    result = backtracking_coloring(g)
    assert result.num_colors == 5
    assert _is_proper_coloring(g, result.coloring)


def test_even_cycle_is_bipartite():
    g = cycle_graph(6)
    result = backtracking_coloring(g)
    assert result.num_colors == 2
    assert _is_proper_coloring(g, result.coloring)


def test_odd_cycle_needs_three_colors():
    g = cycle_graph(5)
    result = backtracking_coloring(g)
    assert result.num_colors == 3
    assert _is_proper_coloring(g, result.coloring)


def test_greedy_coloring_is_always_proper():
    g = complete_graph(6)
    result = greedy_coloring(g)
    assert _is_proper_coloring(g, result.coloring)


def test_greedy_uses_at_least_as_many_colors_as_optimal():
    g = cycle_graph(5)
    greedy_result = greedy_coloring(g)
    optimal_result = backtracking_coloring(g)
    assert greedy_result.num_colors >= optimal_result.num_colors


def test_empty_graph_needs_one_color():
    g = Graph(4)
    result = backtracking_coloring(g)
    assert result.num_colors == 1

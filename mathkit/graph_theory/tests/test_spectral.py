"""Tests for spectral graph theory against known/closed-form Laplacian
spectra."""

import numpy as np
import pytest

from mathkit.graph_theory.systems.spectral import spectral_analysis
from mathkit.graph_theory.utils.generators import complete_graph, cycle_graph, path_graph


def test_four_cycle_matches_known_eigenvalues():
    g = cycle_graph(4)
    result = spectral_analysis(g)
    np.testing.assert_allclose(sorted(result.eigenvalues), [0.0, 2.0, 2.0, 4.0], atol=1e-8)


def test_complete_graph_eigenvalues():
    """K_n has Laplacian eigenvalues 0 (once) and n (n-1 times)."""
    n = 5
    g = complete_graph(n)
    result = spectral_analysis(g)
    eigenvalues = sorted(result.eigenvalues)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(eigenvalues[1:], [n] * (n - 1), atol=1e-8)


def test_smallest_eigenvalue_is_always_zero():
    g = path_graph(6)
    result = spectral_analysis(g)
    assert result.eigenvalues[0] == pytest.approx(0.0, abs=1e-8)


def test_algebraic_connectivity_is_second_smallest_eigenvalue():
    g = path_graph(6)
    result = spectral_analysis(g)
    assert result.algebraic_connectivity == pytest.approx(sorted(result.eigenvalues)[1])


def test_disconnected_graph_has_zero_algebraic_connectivity():
    from mathkit.graph_theory.core.base import Graph

    g = Graph(4)  # no edges at all: 4 isolated vertices
    result = spectral_analysis(g)
    assert result.algebraic_connectivity == pytest.approx(0.0, abs=1e-8)


def test_bipartition_has_both_sides_nonempty_for_connected_graph():
    g = path_graph(6)
    result = spectral_analysis(g)
    assert 0 < np.sum(result.bipartition) < g.n_vertices

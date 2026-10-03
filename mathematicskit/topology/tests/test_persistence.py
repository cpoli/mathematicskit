"""Tests for mathematicskit.topology.systems.point_clouds and persistence."""

import numpy as np
import pytest

from mathematicskit.topology import (
    Filtration,
    SimplicialComplex,
    betti_numbers,
    bottleneck_distance,
    cech_complex,
    circle,
    clique_simplices,
    lower_star_filtration,
    mapper_graph,
    persistent_homology,
    planar_betti_numbers,
    torus,
    vietoris_rips_complex,
    vietoris_rips_filtration,
)


def _noisy_circle(n, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, noise, (n, 2))


def test_rips_complex_of_a_regular_hexagon():
    t = 2 * np.pi * np.arange(6) / 6
    hexagon = np.column_stack([np.cos(t), np.sin(t)])
    assert vietoris_rips_complex(hexagon, 0.51).f_vector == (6, 6)
    assert betti_numbers(vietoris_rips_complex(hexagon, 0.51)) == (1, 1)
    assert betti_numbers(vietoris_rips_complex(hexagon, 0.9, max_dim=3))[:2] == (1, 0)
    assert vietoris_rips_complex(hexagon, 0.4).f_vector == (6,)


def test_cech_is_inside_rips_and_satisfies_the_nerve_theorem():
    X = _noisy_circle(40, seed=3)
    r = 0.3
    C, R = cech_complex(X, r), vietoris_rips_complex(X, r)
    assert set(C.simplices()) <= set(R.simplices())
    xs = np.linspace(-1.5, 1.5, 500)
    gx, gy = np.meshgrid(xs, xs)
    union = np.zeros_like(gx, dtype=bool)
    for p in X:
        union |= np.hypot(gx - p[0], gy - p[1]) <= r
    assert betti_numbers(C)[:2] == planar_betti_numbers(union)
    with pytest.raises(ValueError):
        cech_complex(np.zeros((3, 3)), 1.0)


def test_rips_at_r_is_inside_cech_at_jung_radius():
    X = np.random.default_rng(2).uniform(size=(25, 2))
    r = 0.15
    assert set(vietoris_rips_complex(X, r).simplices()) <= set(cech_complex(X, 2 * r / np.sqrt(3)).simplices())


def test_clique_enumeration_of_complete_graph():
    cliques = clique_simplices(np.ones((5, 5), dtype=bool), 2)
    assert len(cliques) == 5 + 10 + 10


def test_filtration_is_sorted_and_complexes_are_nested():
    F = Filtration(simplices=[(0, 1), (1,), (0,)], values=[1.0, 0.0, 0.0])
    assert F.simplices == [(0,), (1,), (0, 1)] and len(F) == 3
    assert F.complex_at(0.5).f_vector == (2,)


def test_persistence_of_noisy_circle_has_one_long_h1_bar():
    X = _noisy_circle(60)
    diagram = persistent_homology(vietoris_rips_filtration(X, max_dim=2), max_dim=1)
    lifetimes = np.sort(diagram.persistence(1))[::-1]
    assert lifetimes[0] > 0.5 and np.all(lifetimes[1:] < 0.15)
    assert np.sum(np.isinf(diagram.diagram(0)[:, 1])) == 1
    birth, death = diagram.diagram(1)[np.argmax(diagram.persistence(1))]
    assert diagram.betti_numbers((birth + death) / 2) == (1, 1)


def test_persistence_betti_numbers_match_the_complex_at_each_value():
    X = _noisy_circle(30, seed=4, noise=0.1)
    F = vietoris_rips_filtration(X, max_dim=2)
    diagram = persistent_homology(F, max_dim=1, include_zero=True)
    for t in (0.05, 0.2, 0.4, 0.8):
        assert diagram.betti_numbers(t) == betti_numbers(F.complex_at(t))[:2]


def test_lower_star_persistence_of_a_closed_surface_reports_top_class():
    T = torus(6, 4)
    diagram = persistent_homology(lower_star_filtration(T, T.coordinates[:, 2]))
    essential = [int(np.sum(np.isinf(diagram.diagram(k)[:, 1]))) for k in range(3)]
    assert essential == [1, 2, 1]


def test_lower_star_persistence_of_a_function_on_a_path_pairs_minima_with_maxima():
    f = np.array([0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    path = SimplicialComplex([[i, i + 1] for i in range(5)])
    diagram = persistent_homology(lower_star_filtration(path, f))
    assert diagram.diagram(0).tolist() == [[0.0, np.inf], [1.0, 3.0], [2.0, 4.0]]
    assert persistent_homology(Filtration([], [])).pairs == {}


@pytest.mark.parametrize("seed", range(4))
def test_stability_bottleneck_distance_is_at_most_sup_norm_perturbation(seed):
    rng = np.random.default_rng(seed)
    path = SimplicialComplex([[i, i + 1] for i in range(149)])
    f = np.sin(np.linspace(0, 5 * np.pi, 150)) + 0.3 * np.sin(np.linspace(0, 23, 150))
    g = f + rng.uniform(-0.1, 0.1, 150)
    d0 = bottleneck_distance(persistent_homology(lower_star_filtration(path, f)).diagram(0), persistent_homology(lower_star_filtration(path, g)).diagram(0))
    assert d0 <= np.abs(f - g).max() + 1e-12


def test_bottleneck_distance_basic_cases():
    assert bottleneck_distance([[0, 1]], [[0, 1]]) == 0.0
    assert bottleneck_distance([[0, 2]], []) == 1.0
    assert bottleneck_distance([], []) == 0.0
    assert bottleneck_distance([[0, np.inf]], []) == np.inf
    assert bottleneck_distance([[0, np.inf], [0, 1]], [[0.25, np.inf]]) == 0.5
    # (0, 1.1) moves 0.05 onto (0.05, 1.05); (0, 1) goes to the diagonal at distance 0.5
    assert bottleneck_distance([[0, 1], [0, 1.1]], [[0.05, 1.05]]) == pytest.approx(0.5)


def test_mapper_graph_of_noisy_circle_is_a_cycle():
    X = _noisy_circle(300)
    result = mapper_graph(X, X[:, 0], eps=0.25, n_intervals=8, overlap=0.4)
    assert betti_numbers(result.graph) == (1, 1)
    assert sum(len(m) for m in result.nodes) >= 300
    assert len(result.node_values) == len(result.nodes)


def test_mapper_graph_of_two_blobs_has_two_components():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal((-3, 0), 0.3, (50, 2)), rng.normal((3, 0), 0.3, (50, 2))])
    result = mapper_graph(X, X[:, 1], eps=1.0, n_intervals=4)
    assert betti_numbers(result.graph)[0] == 2
    single = mapper_graph(np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([0.0, 1.0]), eps=0.5, n_intervals=3, overlap=0.0)
    assert len(single.nodes) == 2


def test_circle_is_the_one_skeleton_used_by_persistence():
    diagram = persistent_homology(lower_star_filtration(circle(6), np.arange(6.0)))
    assert diagram.diagram(1).tolist() == [[5.0, np.inf]]

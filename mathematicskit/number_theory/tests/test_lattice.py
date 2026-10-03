"""Tests for LLL lattice reduction and integer relations."""

import math
from fractions import Fraction

import numpy as np
import pytest

from mathematicskit.number_theory.systems.lattice import integer_relation, lll_reduce


def _is_lll_reduced(basis, delta=Fraction(3, 4)):
    from mathematicskit.number_theory.systems.lattice import _gram_schmidt

    mu, norms = _gram_schmidt(basis)
    n = len(basis)
    size = all(abs(mu[i][j]) <= Fraction(1, 2) for i in range(n) for j in range(i))
    lovasz = all(norms[k] >= (delta - mu[k][k - 1] ** 2) * norms[k - 1] for k in range(1, n))
    return size and lovasz


def test_reduced_basis_spans_same_lattice():
    rng = np.random.default_rng(0)
    for _ in range(5):
        basis = rng.integers(-50, 50, size=(4, 4)).tolist()
        if round(np.linalg.det(basis)) == 0:
            continue
        reduced = lll_reduce(basis).basis
        assert _is_lll_reduced(reduced)
        assert abs(round(np.linalg.det(reduced))) == abs(round(np.linalg.det(basis)))  # same covolume
        transform = np.linalg.solve(np.array(basis, dtype=float).T, np.array(reduced, dtype=float).T)
        assert np.allclose(transform, np.round(transform), atol=1e-8)  # unimodular change of basis


def test_finds_hidden_short_vector():
    # A unimodular scramble of the standard basis reduces back to unit vectors.
    u = np.array([[1, 0, 0], [7, 1, 0], [-3, 12, 1]]) @ np.array([[1, 5, -2], [0, 1, 9], [0, 0, 1]])
    reduced = lll_reduce(u.tolist())
    assert sorted(sorted(abs(c) for c in row) for row in reduced.basis) == [[0, 0, 1]] * 3
    assert reduced.swaps > 0


def test_first_vector_within_lll_bound():
    rng = np.random.default_rng(1)
    basis = rng.integers(-1000, 1000, size=(5, 5)).tolist()
    reduced = lll_reduce(basis)
    shortest_gs = min(reduced.gram_schmidt_norms)  # lambda_1^2 >= min ||b_i*||^2
    assert sum(c * c for c in reduced.basis[0]) <= 2 ** (5 - 1) * shortest_gs + 1e-9


def test_rejects_dependent_vectors():
    with pytest.raises(ValueError):
        lll_reduce([[1, 2], [2, 4]])


def test_integer_relations():
    assert integer_relation([1.0, math.sqrt(2), 2.0]) in ([2, 0, -1], [-2, 0, 1])
    phi = (1 + math.sqrt(5)) / 2
    assert integer_relation([1.0, phi, phi**2]) == [1, 1, -1]  # phi^2 = phi + 1
    alpha = 2 ** (1 / 3)
    assert integer_relation([alpha**k for k in range(4)], scale=1e10) == [2, 0, 0, -1]

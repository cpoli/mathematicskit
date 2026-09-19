"""Tests for SVD (numpy.linalg.svd) against closed-form/known results."""

import numpy as np

from mathkit.linalg.systems.svd import svd_decompose


def test_svd_reconstructs_diagonal_matrix():
    a = np.array([[3.0, 0.0], [0.0, -2.0]])
    result = svd_decompose(a)
    np.testing.assert_allclose(sorted(result.S, reverse=True), [3.0, 2.0], atol=1e-8)
    np.testing.assert_allclose(result.U @ np.diag(result.S) @ result.Vt, a, atol=1e-8)


def test_svd_reconstructs_general_matrix():
    rng = np.random.default_rng(13)
    a = rng.uniform(-3, 3, size=(5, 3))
    result = svd_decompose(a)
    np.testing.assert_allclose(result.U @ np.diag(result.S) @ result.Vt, a, atol=1e-7)


def test_singular_values_match_numpy_full():
    rng = np.random.default_rng(14)
    a = rng.uniform(-3, 3, size=(6, 4))
    result = svd_decompose(a)
    expected = np.linalg.svd(a, compute_uv=False)
    np.testing.assert_allclose(np.sort(result.S)[::-1], np.sort(expected)[::-1], atol=1e-6)


def test_right_singular_vectors_are_orthonormal():
    rng = np.random.default_rng(15)
    a = rng.uniform(-1, 1, size=(4, 4))
    result = svd_decompose(a)
    v = result.Vt.T
    np.testing.assert_allclose(v.T @ v, np.eye(4), atol=1e-7)


def test_singular_values_are_descending():
    rng = np.random.default_rng(16)
    a = rng.uniform(-2, 2, size=(5, 5))
    result = svd_decompose(a)
    assert np.all(np.diff(result.S) <= 1e-12)

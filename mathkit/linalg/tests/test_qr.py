"""Tests for Householder and Gram-Schmidt QR decomposition."""

import numpy as np
import pytest

from mathkit.linalg.systems.qr import gram_schmidt_qr, householder_qr, orthogonality_error


def test_householder_reconstructs_matrix_and_is_orthogonal():
    a = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    result = householder_qr(a)
    np.testing.assert_allclose(result.Q @ result.R, a, atol=1e-10)
    np.testing.assert_allclose(result.Q.T @ result.Q, np.eye(2), atol=1e-10)


def test_gram_schmidt_reconstructs_matrix():
    a = np.array([[1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    for modified in (True, False):
        result = gram_schmidt_qr(a, modified=modified)
        np.testing.assert_allclose(result.Q @ result.R, a, atol=1e-8)


def test_householder_and_gram_schmidt_agree_on_well_conditioned_matrix():
    rng = np.random.default_rng(5)
    a = rng.uniform(-2, 2, size=(6, 4))
    qr_h = householder_qr(a)
    qr_g = gram_schmidt_qr(a)
    # R can differ in sign per column; compare |R| diagonal magnitude instead.
    np.testing.assert_allclose(np.abs(np.diag(qr_h.R)), np.abs(np.diag(qr_g.R)), rtol=1e-6)


def test_modified_gram_schmidt_more_orthogonal_than_classical_for_ill_conditioned_matrix():
    """The textbook demonstration (Trefethen & Bau, Lecture 9): for a
    nearly-collinear-column matrix, classical Gram-Schmidt loses
    orthogonality far more than modified Gram-Schmidt."""
    eps = 1e-8
    a = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
    q_classical = gram_schmidt_qr(a, modified=False).Q
    q_modified = gram_schmidt_qr(a, modified=True).Q
    assert orthogonality_error(q_modified) < orthogonality_error(q_classical)


def test_householder_stays_orthogonal_for_ill_conditioned_matrix():
    eps = 1e-8
    a = np.array([[1.0, 1.0, 1.0], [eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]])
    q_householder = householder_qr(a).Q
    assert orthogonality_error(q_householder) < 1e-8


def test_qr_rejects_wide_matrix():
    with pytest.raises(ValueError):
        householder_qr(np.ones((2, 3)))
    with pytest.raises(ValueError):
        gram_schmidt_qr(np.ones((2, 3)))

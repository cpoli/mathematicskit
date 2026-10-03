"""Tests for principal component analysis."""

import numpy as np
import pytest

from mathematicskit.statistics.systems.pca import principal_component_analysis


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(0)
    cov = np.array([[4.0, 1.5, 0.5], [1.5, 2.0, 0.3], [0.5, 0.3, 1.0]])
    return rng.multivariate_normal([1.0, -2.0, 3.0], cov, size=5000)


def test_components_are_covariance_eigenvectors(correlated_data):
    result = principal_component_analysis(correlated_data)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(correlated_data, rowvar=False))
    assert np.allclose(result.explained_variance, eigenvalues[::-1])
    for k in range(3):
        assert abs(result.components[k] @ eigenvectors[:, 2 - k]) == pytest.approx(1.0)


def test_components_orthonormal_and_scores_uncorrelated(correlated_data):
    result = principal_component_analysis(correlated_data)
    assert np.allclose(result.components @ result.components.T, np.eye(3))
    assert np.allclose(np.cov(result.scores, rowvar=False), np.diag(result.explained_variance))
    assert result.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert np.allclose(result.mean, correlated_data.mean(axis=0))


def test_reconstruction_from_all_components_is_exact(correlated_data):
    result = principal_component_analysis(correlated_data)
    assert np.allclose(result.scores @ result.components + result.mean, correlated_data)


def test_truncation_minimizes_squared_distance_pearson(correlated_data):
    # Pearson's best-fitting plane: the residual sum of squares equals the discarded variance times (n - 1).
    result = principal_component_analysis(correlated_data, n_components=2)
    assert result.components.shape == (2, 3) and result.scores.shape == (5000, 2)
    residual = correlated_data - (result.scores @ result.components + result.mean)
    full = principal_component_analysis(correlated_data)
    assert np.sum(residual**2) == pytest.approx(full.explained_variance[2] * (len(correlated_data) - 1))


def test_standardize_uses_correlation_matrix(correlated_data):
    result = principal_component_analysis(correlated_data, standardize=True)
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(correlated_data, rowvar=False))
    assert np.allclose(result.explained_variance, eigenvalues[::-1])
    assert result.explained_variance.sum() == pytest.approx(3.0)


def test_sign_convention_and_validation():
    result = principal_component_analysis(np.array([[0.0, 0.0], [-1.0, -2.0], [1.0, 2.0]]))
    assert result.components[0][np.argmax(np.abs(result.components[0]))] > 0
    with pytest.raises(ValueError):
        principal_component_analysis(np.ones(5))

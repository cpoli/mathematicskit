r"""Principal component analysis, via :func:`numpy.linalg.svd`.

Pearson (1901) asked for the line or plane of closest fit to a cloud of
points; Hotelling (1933) asked for the uncorrelated linear combinations
of variables with the largest variance. Both lead to the eigenvectors of
the sample covariance matrix. Computing them as the right singular
vectors of the centered data matrix :math:`X_c = U \Sigma V^T` avoids
forming :math:`X_c^T X_c`, which squares the condition number. See K.
Pearson, "On Lines and Planes of Closest Fit to Systems of Points in
Space," Philosophical Magazine 2(11) (1901), 559-572; H. Hotelling,
"Analysis of a Complex of Statistical Variables into Principal
Components," Journal of Educational Psychology 24(6) (1933), 417-441;
and Jolliffe, *Principal Component Analysis*, 2nd ed. (2002), Ch. 1-3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mathematicskit.statistics.core.base import PCAResult

__all__ = ["principal_component_analysis"]


def principal_component_analysis(data: np.ndarray, n_components: Optional[int] = None, standardize: bool = False) -> PCAResult:
    r"""Principal components of a data matrix (rows are observations).

    With the centered data :math:`X_c = U\Sigma V^T`, the components are
    the rows of :math:`V^T`, the variance along component :math:`k` is
    :math:`\sigma_k^2/(n-1)`, and the scores are :math:`U\Sigma = X_c V`.
    The first :math:`k` components span Pearson's best-fitting
    :math:`k`-dimensional subspace (least total squared perpendicular
    distance). Each component's sign is fixed so that its largest-magnitude
    entry is positive.

    Parameters
    ----------
    data : array-like, shape (n, p)
    n_components : int, optional
        Number of components kept; defaults to ``min(n, p)``.
    standardize : bool
        Divide each column by its standard deviation first, i.e. use the
        correlation matrix instead of the covariance matrix (appropriate
        when the variables have different units).

    Returns
    -------
    PCAResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((2000, 2)) * [3.0, 1.0] @ np.array([[0.6, 0.8], [-0.8, 0.6]])
    >>> result = principal_component_analysis(data)
    >>> np.round(result.components[0], 1)
    array([0.6, 0.8])
    >>> bool(abs(result.explained_variance[0] - 9.0) < 0.5)
    True
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_observations, n_variables)")
    n = x.shape[0]
    mean = x.mean(axis=0)
    centered = x - mean
    if standardize:
        centered = centered / x.std(axis=0, ddof=1)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    k = vt.shape[0] if n_components is None else int(n_components)
    signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
    vt = vt * signs[:, None]
    variance = singular_values**2 / (n - 1)
    total = variance.sum()
    return PCAResult(
        components=vt[:k],
        explained_variance=variance[:k],
        explained_variance_ratio=variance[:k] / total if total > 0 else np.zeros(k),
        scores=centered @ vt[:k].T,
        mean=mean,
        singular_values=singular_values[:k],
    )

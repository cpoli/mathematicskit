"""Gershgorin's circle theorem: eigenvalue inclusion discs read directly off a matrix.

Hand-rolled: the discs are a closed-form function of the matrix entries
(no numpy/scipy routine computes them). Every eigenvalue of ``A`` lies in
the union of the row discs :math:`\\{z : |z - a_{kk}| \\le \\sum_{j \\ne k} |a_{kj}|\\}`,
and -- since ``A`` and ``A^T`` share eigenvalues -- also in the union of
the column discs. See S. Gerschgorin, "Über die Abgrenzung der
Eigenwerte einer Matrix," Izv. Akad. Nauk SSSR 6 (1931), 749-754, and
R. S. Varga, *Geršgorin and His Circles*, Springer, 2004, Ch. 1.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.linalg.core.base import GershgorinResult

__all__ = ["gershgorin_discs"]


def gershgorin_discs(a: np.ndarray, by: str = "rows") -> GershgorinResult:
    r"""The Gershgorin discs of a square matrix.

    Disc :math:`k` has center :math:`a_{kk}` and radius
    :math:`R_k = \sum_{j \ne k} |a_{kj}|` (``by="rows"``) or
    :math:`\sum_{j \ne k} |a_{jk}|` (``by="columns"``). A consequence:
    a strictly diagonally dominant matrix (every :math:`|a_{kk}| > R_k`)
    has no eigenvalue at 0 and is therefore nonsingular.

    Parameters
    ----------
    a : ndarray, shape (n, n)
        Real or complex square matrix.
    by : {"rows", "columns"}

    Returns
    -------
    GershgorinResult

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[10.0, 1.0, 0.0], [0.2, 8.0, 0.2], [1.0, 1.0, 2.0]])
    >>> discs = gershgorin_discs(A)
    >>> discs.centers, discs.radii
    (array([10.,  8.,  2.]), array([1. , 0.4, 2. ]))
    >>> all(discs.contains(lam) for lam in np.linalg.eigvals(A))
    True
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")
    if by == "columns":
        a = a.T
    elif by != "rows":
        raise ValueError("by must be 'rows' or 'columns'")
    centers = np.diag(a).copy()
    radii = np.sum(np.abs(a), axis=1) - np.abs(centers)
    return GershgorinResult(centers=centers, radii=radii.astype(np.float64))

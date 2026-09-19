"""Small matrix-generation and norm helpers used across mathkit.linalg's
systems/ modules and tests -- supporting numerics, not models themselves.
"""

from __future__ import annotations

import numpy as np

__all__ = ["random_spd_matrix", "frobenius_norm", "is_symmetric"]


def random_spd_matrix(n: int, seed: int = 0, condition_scale: float = 1.0) -> np.ndarray:
    r"""Generate a random symmetric positive-definite matrix.

    Constructs :math:`A = M^T M + \epsilon I` for a random :math:`M`,
    which is SPD for any nonsingular :math:`M` (:math:`x^T A x = \|Mx\|^2
    + \epsilon\|x\|^2 > 0` for :math:`x \neq 0`); `condition_scale` scales
    :math:`M`'s entries to make the resulting matrix better- or
    worse-conditioned for stability demonstrations.

    Parameters
    ----------
    n : int
        Matrix size.
    seed : int
        Random seed, for reproducibility.
    condition_scale : float
        Multiplies the generating matrix's entries.

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> import numpy as np
    >>> A = random_spd_matrix(4, seed=1)
    >>> np.allclose(A, A.T)
    True
    >>> bool(np.all(np.linalg.eigvalsh(A) > 0))
    True
    """
    rng = np.random.default_rng(seed)
    m = condition_scale * rng.normal(size=(n, n))
    return m.T @ m + 0.1 * np.eye(n)


def frobenius_norm(a: np.ndarray) -> float:
    r"""Frobenius norm :math:`\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}`, via :func:`numpy.linalg.norm`.

    Parameters
    ----------
    a : ndarray

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> round(frobenius_norm(np.array([[3.0, 0.0], [0.0, 4.0]])), 6)
    5.0
    """
    a = np.asarray(a, dtype=np.float64)
    return float(np.linalg.norm(a, ord="fro"))


def is_symmetric(a: np.ndarray, atol: float = 1e-10) -> bool:
    """Check whether a matrix is (numerically) symmetric.

    Parameters
    ----------
    a : ndarray, shape (n, n)
    atol : float

    Returns
    -------
    bool

    Examples
    --------
    >>> import numpy as np
    >>> is_symmetric(np.array([[1.0, 2.0], [2.0, 1.0]]))
    True
    """
    a = np.asarray(a, dtype=np.float64)
    return bool(np.allclose(a, a.T, atol=atol))

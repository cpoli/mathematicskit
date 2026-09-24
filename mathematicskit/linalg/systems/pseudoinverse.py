"""The Moore-Penrose pseudoinverse and its four defining equations.

:func:`pseudoinverse` is a thin wrapper around :func:`numpy.linalg.pinv`
(SVD-based: invert the singular values above a cutoff, zero the rest).
:func:`penrose_residuals` checks Penrose's four equations, which
characterize :math:`A^+` uniquely. See E. H. Moore, Bull. AMS 26 (1920),
394-395; R. Penrose, "A Generalized Inverse for Matrices," Proc.
Cambridge Philos. Soc. 51 (1955), 406-413; and Golub & Van Loan,
*Matrix Computations*, 4th ed., Ch. 5.5.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["pseudoinverse", "penrose_residuals"]


def pseudoinverse(a: np.ndarray, rcond: Optional[float] = None) -> np.ndarray:
    r"""Moore-Penrose pseudoinverse :math:`A^+ = V \Sigma^+ U^T`, via :func:`numpy.linalg.pinv`.

    For any right-hand side ``b``, :math:`x = A^+ b` is the
    minimum-norm least-squares solution of ``A x = b`` -- it exists and is
    unique even when ``A`` is rectangular or rank-deficient.

    Parameters
    ----------
    a : ndarray, shape (m, n)
    rcond : float, optional
        Singular values below ``rcond * max(S)`` are treated as zero;
        defaults to numpy's ``max(m, n) * eps``.

    Returns
    -------
    ndarray, shape (n, m)

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    >>> pseudoinverse(A)
    array([[0.25, 0.25],
           [0.25, 0.25]])
    """
    a = np.asarray(a)
    if rcond is None:
        return np.linalg.pinv(a)
    return np.linalg.pinv(a, rcond=rcond)


def penrose_residuals(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""Frobenius-norm residuals of Penrose's four equations for a candidate ``X = A^+``.

    .. math::

       AXA = A, \qquad XAX = X, \qquad (AX)^H = AX, \qquad (XA)^H = XA.

    Penrose (1955) proved exactly one ``X`` satisfies all four.

    Parameters
    ----------
    a : ndarray, shape (m, n)
    x : ndarray, shape (n, m)

    Returns
    -------
    ndarray, shape (4,)
        ``||AXA - A||_F``, ``||XAX - X||_F``, ``||(AX)^H - AX||_F``,
        ``||(XA)^H - XA||_F``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 2.0], [2.0, 4.0], [0.0, 1.0]])
    >>> bool(np.all(penrose_residuals(A, pseudoinverse(A)) < 1e-12))
    True
    """
    a = np.asarray(a)
    x = np.asarray(x)
    ax = a @ x
    xa = x @ a
    return np.array(
        [
            np.linalg.norm(ax @ a - a),
            np.linalg.norm(xa @ x - x),
            np.linalg.norm(ax.conj().T - ax),
            np.linalg.norm(xa.conj().T - xa),
        ]
    )

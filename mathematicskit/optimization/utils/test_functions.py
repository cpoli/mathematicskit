r"""Shared benchmark objective functions for
mathematicskit.optimization's convergence-rate comparisons -- not models in
their own right, but the common "test track" every method in
:mod:`mathematicskit.optimization.systems` is compared on.

See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 2 (the
Rosenbrock function, eq. 2.22, as the standard "banana valley" torture
test) and any convex-optimization text for the quadratic bowl.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["rosenbrock", "rosenbrock_grad", "rosenbrock_hess", "quadratic_bowl", "quadratic_bowl_grad", "quadratic_bowl_hess"]

_DEFAULT_BOWL_DIAGONAL = (1.0, 10.0)


def _bowl_matrix(n: int, matrix) -> np.ndarray:
    """Resolve the quadratic bowl's matrix ``A``, defaulting to ``diag(1, 10)``.

    The default is two-dimensional, so anything past ``n == 2`` has to supply
    its own ``matrix``; saying so here beats letting the mismatch surface as
    an opaque ``matmul`` shape error from deep inside the objective.
    """
    if matrix is not None:
        a = np.asarray(matrix, dtype=np.float64)
        if a.shape != (n, n):
            raise ValueError(f"matrix must have shape ({n}, {n}) to match x, got {a.shape}")
        return a
    if n > len(_DEFAULT_BOWL_DIAGONAL):
        raise ValueError(f"the default quadratic bowl is 2-dimensional; pass an explicit (matrix) of shape ({n}, {n}) for x of length {n}")
    return np.diag(np.asarray(_DEFAULT_BOWL_DIAGONAL[:n], dtype=np.float64))


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    r"""The (generalized, n-dimensional) Rosenbrock "banana" function.

    :math:`f(x) = \sum_{i=1}^{n-1} \left[b(x_{i+1}-x_i^2)^2 +
    (a-x_i)^2\right]`. A narrow, curved valley with a unique global
    minimum :math:`f(a, a, \dots, a) = 0`; gradient descent zig-zags
    slowly along the valley floor while Newton/quasi-Newton methods that
    exploit curvature converge far faster -- the standard demonstration
    of why second-order information matters. See Nocedal & Wright,
    *Numerical Optimization*, 2nd ed., eq. 2.22.

    Parameters
    ----------
    x : ndarray, shape (n,), n >= 2
    a, b : float
        Standard values are ``a=1``, ``b=100``.

    Returns
    -------
    float

    Examples
    --------
    >>> rosenbrock(np.array([1.0, 1.0]))
    0.0
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2))


def rosenbrock_grad(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    r"""Gradient of :func:`rosenbrock`.

    Parameters
    ----------
    x : ndarray, shape (n,), n >= 2
    a, b : float

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> np.allclose(rosenbrock_grad(np.array([1.0, 1.0])), 0.0)
    True
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    xi, xi1 = x[:-1], x[1:]
    grad[:-1] += -4.0 * b * (xi1 - xi**2) * xi - 2.0 * (a - xi)
    grad[1:] += 2.0 * b * (xi1 - xi**2)
    return grad


def rosenbrock_hess(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    r"""Hessian of :func:`rosenbrock` (tridiagonal).

    Parameters
    ----------
    x : ndarray, shape (n,), n >= 2
    a, b : float

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> H = rosenbrock_hess(np.array([1.0, 1.0]))
    >>> H.shape
    (2, 2)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    hess = np.zeros((n, n))
    for i in range(n - 1):
        hess[i, i] += 12.0 * b * x[i] ** 2 - 4.0 * b * x[i + 1] + 2.0
        hess[i, i + 1] += -4.0 * b * x[i]
        hess[i + 1, i] += -4.0 * b * x[i]
        hess[i + 1, i + 1] += 2.0 * b
    return hess


def quadratic_bowl(x: np.ndarray, matrix: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None) -> float:
    r"""A convex quadratic bowl :math:`f(x) = \tfrac12 x^T A x - b^T x`.

    For symmetric positive-definite ``A``, the unique minimizer is
    :math:`x^* = A^{-1}b`, so this is the standard convex test problem
    (any descent method converges, but at rates governed by
    :math:`\kappa(A)`, the condition number).

    Parameters
    ----------
    x : ndarray, shape (n,)
    matrix : ndarray, shape (n, n), optional
        SPD matrix ``A``; defaults to ``diag([1, 10])`` for ``n=2`` (a
        mildly ill-conditioned bowl -- elongated contours -- that makes
        gradient descent's zig-zagging visible).
    b : ndarray, shape (n,), optional
        Defaults to the zero vector (minimizer at the origin).

    Returns
    -------
    float

    Examples
    --------
    >>> round(quadratic_bowl(np.array([0.0, 0.0])), 10)
    0.0
    """
    x = np.asarray(x, dtype=np.float64)
    a = _bowl_matrix(x.shape[0], matrix)
    bb = np.zeros_like(x) if b is None else np.asarray(b, dtype=np.float64)
    return float(0.5 * x @ a @ x - bb @ x)


def quadratic_bowl_grad(x: np.ndarray, matrix: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Gradient of :func:`quadratic_bowl`: :math:`\nabla f(x) = Ax - b`.

    Parameters
    ----------
    x : ndarray, shape (n,)
    matrix : ndarray, shape (n, n), optional
    b : ndarray, shape (n,), optional

    Returns
    -------
    ndarray, shape (n,)
    """
    x = np.asarray(x, dtype=np.float64)
    a = _bowl_matrix(x.shape[0], matrix)
    bb = np.zeros_like(x) if b is None else np.asarray(b, dtype=np.float64)
    return a @ x - bb


def quadratic_bowl_hess(x: np.ndarray, matrix: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None) -> np.ndarray:
    """Hessian of :func:`quadratic_bowl`: the constant matrix ``A`` itself.

    Parameters
    ----------
    x : ndarray, shape (n,)
    matrix : ndarray, shape (n, n), optional
    b : ndarray, shape (n,), optional
        Unused; accepted for a uniform ``(x, matrix, b)`` signature
        alongside :func:`quadratic_bowl`/:func:`quadratic_bowl_grad`.

    Returns
    -------
    ndarray, shape (n, n)
    """
    x = np.asarray(x, dtype=np.float64)
    return _bowl_matrix(x.shape[0], matrix)

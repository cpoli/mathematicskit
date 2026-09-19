r"""Fixed-point stability analysis for 2D autonomous systems via Jacobian
linearization: the trace-determinant classification of node/saddle/
spiral/center, and a multivariate Newton solver (built on
:mod:`mathematicskit.linalg`) to locate fixed points numerically.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 5
("Linear Systems") and Ch. 6.3 ("Linearization"), and Burden & Faires,
*Numerical Analysis*, 10th ed., Ch. 10.2 (multivariate Newton's method).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.linalg.systems.lu import lu_solve_system
from mathematicskit.ode_dynamics.core.base import FixedPointResult

__all__ = ["classify_fixed_point_2d", "find_fixed_point_newton", "numerical_jacobian"]


def classify_fixed_point_2d(jacobian: np.ndarray, location=None) -> FixedPointResult:
    r"""Classify a 2D linear system's fixed point from its Jacobian.

    For :math:`\dot{\mathbf x} = J\mathbf x` (or the linearization of a
    nonlinear system about a fixed point), let :math:`\tau =
    \mathrm{tr}(J)` and :math:`\Delta = \det(J)`; the eigenvalues are
    :math:`\lambda_{1,2} = (\tau \pm \sqrt{\tau^2 - 4\Delta})/2`. Then:

    - :math:`\Delta < 0`: saddle (always unstable).
    - :math:`\Delta > 0`, :math:`\tau^2 > 4\Delta`: node (real
      eigenvalues, same sign as :math:`\tau`).
    - :math:`\Delta > 0`, :math:`\tau^2 < 4\Delta`: spiral (complex
      eigenvalues, real part sign = sign of :math:`\tau`).
    - :math:`\Delta > 0`, :math:`\tau = 0`: center (purely imaginary).
    - :math:`\tau^2 = 4\Delta \neq 0`: degenerate node (repeated real eigenvalue).

    See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 5.2
    (the classification diagram in the :math:`(\tau, \Delta)` plane).

    Parameters
    ----------
    jacobian : ndarray, shape (2, 2)
    location : ndarray, shape (2,), optional
        The fixed point's coordinates, for record-keeping; defaults to
        the origin.

    Returns
    -------
    FixedPointResult

    Examples
    --------
    >>> import numpy as np
    >>> result = classify_fixed_point_2d(np.array([[-1.0, 0.0], [0.0, -2.0]]))
    >>> result.classification
    'stable node'
    >>> result2 = classify_fixed_point_2d(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    >>> result2.classification
    'center'
    """
    j = np.asarray(jacobian, dtype=np.float64)
    tau = float(np.trace(j))
    delta = float(np.linalg.det(j))
    disc = tau * tau - 4.0 * delta
    if disc >= 0:
        sqrt_disc = np.sqrt(disc)
        eigenvalues = np.array([(tau + sqrt_disc) / 2.0, (tau - sqrt_disc) / 2.0], dtype=complex)
    else:
        sqrt_disc = 1j * np.sqrt(-disc)
        eigenvalues = np.array([(tau + sqrt_disc) / 2.0, (tau - sqrt_disc) / 2.0], dtype=complex)

    if delta < 0:
        classification = "saddle"
    elif delta == 0:
        classification = "non-isolated (zero eigenvalue)"
    elif disc > 0:
        if abs(tau * tau - 4.0 * delta) < 1e-12:
            classification = "degenerate node"
        else:
            classification = "stable node" if tau < 0 else "unstable node"
    elif disc < 0:
        if abs(tau) < 1e-12:
            classification = "center"
        else:
            classification = "stable spiral" if tau < 0 else "unstable spiral"
    else:  # disc == 0, delta > 0
        classification = "degenerate node"

    stable = bool(np.all(eigenvalues.real < 0))
    loc = np.zeros(2) if location is None else np.asarray(location, dtype=np.float64)
    return FixedPointResult(location=loc, jacobian=j, eigenvalues=eigenvalues, classification=classification, stable=stable)


def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector field ``f: R^n -> R^n``.

    Parameters
    ----------
    f : callable
    x : ndarray, shape (n,)
    h : float
        Step size.

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1], -x[0]])
    >>> np.round(numerical_jacobian(f, np.array([0.0, 0.0])), 6)
    array([[ 0.,  1.],
           [-1.,  0.]])
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    jac = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = h
        jac[:, j] = (f(x + dx) - f(x - dx)) / (2.0 * h)
    return jac


def find_fixed_point_newton(f: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, tol: float = 1e-10, max_iter: int = 100):
    r"""Locate a fixed point of ``f(x) = 0`` via multivariate Newton's method.

    :math:`x_{k+1} = x_k - J(x_k)^{-1} f(x_k)`, with the linear solve at
    each step done via :func:`mathematicskit.linalg.systems.lu.lu_solve_system`
    (never an explicit matrix inverse) and the Jacobian obtained from
    :func:`numerical_jacobian`. See Burden & Faires, *Numerical
    Analysis*, 10th ed., Ch. 10.2.

    Parameters
    ----------
    f : callable
        Vector field ``f(x) -> ndarray``, same shape as ``x``.
    x0 : ndarray
        Initial guess.
    tol : float
        Convergence tolerance on ``||x_{k+1} - x_k||``.
    max_iter : int

    Returns
    -------
    x : ndarray
        The fixed point found.
    converged : bool

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: np.array([x[1] - x[0]**2, x[0] + x[1] - 2.0])
    >>> x, converged = find_fixed_point_newton(f, np.array([0.5, 0.5]))
    >>> converged
    True
    >>> np.allclose(f(x), 0.0, atol=1e-8)
    True
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    converged = False
    for _ in range(max_iter):
        fx = f(x)
        jac = numerical_jacobian(f, x)
        try:
            delta = lu_solve_system(jac, fx)
        except np.linalg.LinAlgError:
            break
        x_new = x - delta
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            converged = True
            break
        x = x_new
    return x, converged

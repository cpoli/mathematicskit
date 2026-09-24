r"""Fixed-point stability analysis for 2D autonomous systems via Jacobian
linearization: the trace-determinant classification of node/saddle/
spiral/center, and a multivariate Newton solver (built on
:mod:`mathematicskit.linalg`) to locate fixed points numerically.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 5
("Linear Systems") and Ch. 6.3 ("Linearization"), and Burden & Faires,
*Numerical Analysis*, 10th ed., Ch. 10.2 (multivariate Newton's method).
:func:`lyapunov_quadratic_form` implements Lyapunov's (1892) direct
method for linear systems via the continuous Lyapunov equation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from mathematicskit.linalg.systems.lu import lu_solve_system
from mathematicskit.ode_dynamics.core.base import FixedPointResult, LyapunovFunctionResult

__all__ = ["classify_fixed_point_2d", "find_fixed_point_newton", "numerical_jacobian", "lyapunov_quadratic_form"]

_BORDERLINE_TOL = 1e-12
"""Tolerance for the borderline cases in :func:`classify_fixed_point_2d`:
a repeated eigenvalue (:math:`\\tau^2 = 4\\Delta`) and a pure center
(:math:`\\tau = 0`). Both are exact-equality conditions that floating-point
arithmetic will essentially never hit on the nose."""


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

    The last two are exact-equality conditions, tested here against a small
    tolerance rather than against zero. A *near*-borderline system is
    genuinely ambiguous rather than merely hard to classify numerically:
    linearization is inconclusive for centers and degenerate nodes, since an
    arbitrarily small nonlinear term can push the trajectory either way
    (Strogatz, Sec. 6.3).

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

    # The borderline cases -- a repeated eigenvalue (disc == 0) and a pure
    # center (tau == 0) -- are exact-equality conditions that floating-point
    # arithmetic essentially never reproduces, so both are tested against a
    # small tolerance rather than against zero. Note that this makes the
    # classification of a near-borderline system genuinely ambiguous, which
    # is the mathematics, not a defect: an arbitrarily small perturbation
    # turns a center into a spiral either way (Strogatz, Sec. 6.3, on why
    # linearization is inconclusive for borderline cases).
    if delta < 0:
        classification = "saddle"
    elif delta == 0:
        classification = "non-isolated (zero eigenvalue)"
    elif abs(disc) < _BORDERLINE_TOL:
        classification = "degenerate node"
    elif disc > 0:
        classification = "stable node" if tau < 0 else "unstable node"
    elif abs(tau) < _BORDERLINE_TOL:
        classification = "center"
    else:
        classification = "stable spiral" if tau < 0 else "unstable spiral"

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


def lyapunov_quadratic_form(A, Q=None) -> LyapunovFunctionResult:
    r"""Quadratic Lyapunov function for the linear system :math:`\dot x = Ax`.

    Lyapunov's direct method (1892) proves stability without solving the
    ODE: if a function :math:`V(x) > 0` (for :math:`x \neq 0`) strictly
    decreases along every trajectory, the origin is asymptotically
    stable. For a linear system, try :math:`V(x) = x^T P x`; then
    :math:`\dot V = x^T(A^T P + P A)x = -x^T Q x`. Solving the
    continuous *Lyapunov equation*

    .. math:: A^T P + P A = -Q

    for a chosen :math:`Q \succ 0` gives a positive-definite :math:`P`
    if and only if every eigenvalue of :math:`A` has negative real part
    (Lyapunov's theorem). The equation is solved with
    :func:`scipy.linalg.solve_continuous_lyapunov` (Bartels-Stewart).
    See Khalil, *Nonlinear Systems*, 3rd ed., Thm. 4.6.

    Parameters
    ----------
    A : array-like, shape (n, n)
        System matrix.
    Q : array-like, shape (n, n), optional
        Symmetric positive-definite matrix; defaults to the identity.

    Returns
    -------
    LyapunovFunctionResult
        ``P``, ``Q``, and whether ``P`` is positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> res = lyapunov_quadratic_form([[-1.0, 0.0], [0.0, -2.0]])
    >>> np.round(res.P, 6)
    array([[0.5 , 0.  ],
           [0.  , 0.25]])
    >>> res.positive_definite
    True
    >>> lyapunov_quadratic_form([[1.0, 0.0], [0.0, -1.0]]).positive_definite
    False
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    Q = np.eye(A.shape[0]) if Q is None else np.asarray(Q, dtype=np.float64)
    if Q.shape != A.shape:
        raise ValueError("Q must have the same shape as A")
    # scipy solves a X + X a^H = q; with a = A^T this is A^T P + P A = -Q.
    P = solve_continuous_lyapunov(A.T, -Q)
    P = 0.5 * (P + P.T)
    positive_definite = bool(np.all(np.linalg.eigvalsh(P) > 0.0))
    return LyapunovFunctionResult(P=P, Q=Q, positive_definite=positive_definite)

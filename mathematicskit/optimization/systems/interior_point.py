r"""Interior-point linear programming: a primal-dual path-following method.

Karmarkar (1984) gave the first polynomial-time LP algorithm that was
also fast in practice. Instead of walking the polytope's edges like the
simplex method, it moves through the interior. Its modern descendants,
the primal-dual methods of Kojima, Mizuno and Yoshise (1989) and
Mehrotra (1992), follow the *central path*: the solutions of the
perturbed optimality conditions

.. math::

   A x = b, \qquad A^T y + s = c, \qquad x_i s_i = \mu, \qquad x, s > 0,

as :math:`\mu \to 0`. Each iteration takes one Newton step towards the
point with target :math:`\sigma\mu` (centring parameter
:math:`\sigma < 1`) and stops short of the boundary. Hand-rolled to
expose that path; :func:`~mathematicskit.optimization.systems.linear_programming.linear_program`
with ``method="highs-ipm"`` is the production solver to check against.
See N. Karmarkar, "A New Polynomial-Time Algorithm for Linear
Programming," Combinatorica 4(4) (1984), 373-395, and S. J. Wright,
*Primal-Dual Interior-Point Methods* (SIAM, 1997), Ch. 1 and 5.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER
from mathematicskit.optimization.core.base import OptimizeResult

__all__ = ["interior_point_lp"]


def _step_to_boundary(v: np.ndarray, dv: np.ndarray) -> float:
    shrinking = dv < 0
    return float(min(1.0, np.min(-v[shrinking] / dv[shrinking]))) if np.any(shrinking) else 1.0


def interior_point_lp(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    sigma: float = 0.1,
    eta: float = 0.95,
    tol: float = 1e-9,
    max_iter: int = DEFAULT_MAX_ITER,
) -> OptimizeResult:
    r"""Solve :math:`\min c^T x` s.t. :math:`A x \leq b`, :math:`x \geq 0` by primal-dual path following.

    Slack variables put the problem in standard form
    :math:`[A\ I]\,(x, w) = b`. From the interior point
    :math:`x = w = s = 1`, :math:`y = 0` (not necessarily feasible),
    each iteration solves the Newton system for the central-path
    equations with target :math:`\sigma\mu`, through the normal
    equations :math:`A D A^T \Delta y = r` with
    :math:`D = \mathrm{diag}(x/s)`, then steps a fraction ``eta`` of the
    way to the boundary of the positive orthant.

    Parameters
    ----------
    c : ndarray, shape (n,)
    a_ub : ndarray, shape (m, n)
    b_ub : ndarray, shape (m,)
    sigma : float
        Centring parameter in ``(0, 1)``: smaller is more aggressive,
        larger stays closer to the central path.
    eta : float
        Fraction of the step to the boundary actually taken, in ``(0, 1)``.
    tol : float
        Stop when the duality measure :math:`\mu = x^T s / N` and the
        relative primal and dual residuals all fall below ``tol``.
    max_iter : int

    Returns
    -------
    OptimizeResult
        ``path`` holds the original variables :math:`x` at every
        iterate, all strictly inside the feasible region once primal
        feasibility is reached. ``extra`` has ``"duality_measure"`` (the
        :math:`\mu` history) and ``"dual"`` (the multipliers :math:`y`
        of the inequality constraints, :math:`\leq 0` at the optimum).

    Examples
    --------
    >>> import numpy as np
    >>> # maximize 3x + 5y s.t. x + 2y <= 40, 2x + y <= 30: optimum (20/3, 50/3).
    >>> result = interior_point_lp(np.array([-3.0, -5.0]), np.array([[1.0, 2.0], [2.0, 1.0]]), np.array([40.0, 30.0]))
    >>> result.converged, np.round(result.x, 6), round(-result.fun, 6)
    (True, array([ 6.666667, 16.666667]), 103.333333)
    """
    c = np.asarray(c, dtype=np.float64)
    a_ub = np.atleast_2d(np.asarray(a_ub, dtype=np.float64))
    b = np.asarray(b_ub, dtype=np.float64)
    m, n = a_ub.shape
    a = np.hstack([a_ub, np.eye(m)])
    cost = np.concatenate([c, np.zeros(m)])
    big_n = n + m
    x, s, y = np.ones(big_n), np.ones(big_n), np.zeros(m)
    path, mus = [x[:n].copy()], []
    converged = False
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        r_primal = a @ x - b
        r_dual = a.T @ y + s - cost
        mu = float(x @ s / big_n)
        mus.append(mu)
        if mu < tol and np.linalg.norm(r_primal) <= tol * (1 + np.linalg.norm(b)) and np.linalg.norm(r_dual) <= tol * (1 + np.linalg.norm(cost)):
            converged = True
            iterations -= 1
            break
        d = x / s
        target = sigma * mu / s
        rhs = -r_primal + a @ (x - target) - a @ (d * r_dual)
        dy = np.linalg.solve((a * d) @ a.T, rhs)
        ds = -r_dual - a.T @ dy
        dx = -x + target - d * ds
        alpha_primal = eta * _step_to_boundary(x, dx)
        alpha_dual = eta * _step_to_boundary(s, ds)
        x = x + alpha_primal * dx
        y = y + alpha_dual * dy
        s = s + alpha_dual * ds
        path.append(x[:n].copy())
    return OptimizeResult(
        x=x[:n],
        fun=float(c @ x[:n]),
        path=np.array(path),
        iterations=iterations,
        converged=converged,
        method="interior_point",
        extra={"duality_measure": np.array(mus), "dual": y},
    )

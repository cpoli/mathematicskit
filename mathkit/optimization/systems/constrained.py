r"""Lagrange multipliers, KKT-condition verification, and a penalty-method
solver for constrained nonlinear optimization.

Equality-constrained stationary points are found by solving the Lagrange
system :math:`\nabla f(x) + \sum_i \lambda_i \nabla h_i(x) = 0`,
:math:`h(x) = 0` directly via :func:`scipy.optimize.root` (a
well-posed square nonlinear system once multipliers are added as
unknowns -- no reason to hand-write a multivariate Newton solver when
mathkit already has one general-purpose root finder available via
scipy). KKT verification at a candidate point is mathkit's own code,
since it's a direct arithmetic check rather than a numerical algorithm
with a library equivalent. See Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Ch. 12 (KKT conditions) and Ch. 17
(penalty/barrier methods).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import optimize as sopt

from mathkit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathkit.optimization.core.base import KKTResult, LagrangeResult, OptimizeResult, UnconstrainedOptimizer
from mathkit.optimization.systems.newton_quasi_newton import BFGS

__all__ = ["lagrange_stationary_point", "verify_kkt", "PenaltyMethod"]


def lagrange_stationary_point(
    grad_f: Callable[[np.ndarray], np.ndarray], h: Callable[[np.ndarray], np.ndarray], grad_h: Callable[[np.ndarray], np.ndarray], x0, lambda0=None
) -> LagrangeResult:
    r"""Solve the equality-constrained Lagrange stationarity system.

    Finds ``(x*, lambda*)`` satisfying :math:`\nabla f(x^*) + J_h(x^*)^T
    \lambda^* = 0` and :math:`h(x^*) = 0` simultaneously, via
    :func:`scipy.optimize.root` on the stacked system -- the standard way
    to *locate* a constrained stationary point once you've written down
    the first-order (Lagrange) conditions symbolically. See Nocedal &
    Wright, *Numerical Optimization*, 2nd ed., Ch. 12.2-12.3.

    Parameters
    ----------
    grad_f : callable
        Gradient of the objective, ``grad_f(x) -> ndarray`` shape (n,).
    h : callable
        Equality constraints, ``h(x) -> ndarray`` shape (m,).
    grad_h : callable
        Constraint Jacobian, ``grad_h(x) -> ndarray`` shape (m, n).
    x0 : ndarray, shape (n,)
        Initial guess for ``x``.
    lambda0 : ndarray, shape (m,), optional
        Initial guess for the multipliers; defaults to zeros.

    Returns
    -------
    LagrangeResult

    Examples
    --------
    >>> import numpy as np
    >>> # minimize x^2 + y^2 subject to x + y = 1: exact solution (0.5, 0.5), lambda=-1.
    >>> grad_f = lambda z: np.array([2.0 * z[0], 2.0 * z[1]])
    >>> h = lambda z: np.array([z[0] + z[1] - 1.0])
    >>> grad_h = lambda z: np.array([[1.0, 1.0]])
    >>> result = lagrange_stationary_point(grad_f, h, grad_h, x0=np.array([0.0, 0.0]))
    >>> np.allclose(result.x, [0.5, 0.5], atol=1e-8)
    True
    >>> np.allclose(result.multipliers, [-1.0], atol=1e-8)
    True
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.shape[0]
    m = np.atleast_1d(h(x0)).shape[0]
    lam0 = np.zeros(m) if lambda0 is None else np.asarray(lambda0, dtype=np.float64)

    def system(z):
        x, lam = z[:n], z[n:]
        stationarity = grad_f(x) + grad_h(x).T @ lam
        feasibility = np.atleast_1d(h(x))
        return np.concatenate([stationarity, feasibility])

    sol = sopt.root(system, np.concatenate([x0, lam0]))
    x_star, lam_star = sol.x[:n], sol.x[n:]
    return LagrangeResult(x=x_star, multipliers=lam_star, converged=bool(sol.success))


def verify_kkt(
    x: np.ndarray,
    grad_f: Callable[[np.ndarray], np.ndarray],
    h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    grad_h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eq_multipliers: Optional[np.ndarray] = None,
    g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    grad_g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ineq_multipliers: Optional[np.ndarray] = None,
    tol: float = 1e-6,
) -> KKTResult:
    r"""Numerically check the KKT conditions at a candidate point.

    Verifies, for equality constraints :math:`h(x)=0` with multipliers
    :math:`\lambda` and inequality constraints :math:`g(x) \leq 0` with
    multipliers :math:`\mu \geq 0`:

    1. **Stationarity**: :math:`\nabla f(x) + \sum_i \lambda_i \nabla
       h_i(x) + \sum_j \mu_j \nabla g_j(x) \approx 0`.
    2. **Primal feasibility**: :math:`h(x) \approx 0`, :math:`g(x) \leq
       \text{tol}`.
    3. **Dual feasibility**: :math:`\mu \geq -\text{tol}`.
    4. **Complementary slackness**: :math:`\mu_j g_j(x) \approx 0` for
       every ``j``.

    See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 12.3,
    Theorem 12.1.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Candidate point.
    grad_f : callable
        Objective gradient.
    h, grad_h : callable, optional
        Equality constraints and their Jacobian.
    eq_multipliers : ndarray, optional
    g, grad_g : callable, optional
        Inequality constraints and their Jacobian.
    ineq_multipliers : ndarray, optional
    tol : float
        Numerical tolerance for each check.

    Returns
    -------
    KKTResult

    Examples
    --------
    >>> import numpy as np
    >>> # minimize x^2 + y^2 subject to x + y = 1: (0.5, 0.5) with lambda=-1 is exactly KKT-stationary.
    >>> grad_f = lambda z: np.array([2.0 * z[0], 2.0 * z[1]])
    >>> h = lambda z: np.array([z[0] + z[1] - 1.0])
    >>> grad_h = lambda z: np.array([[1.0, 1.0]])
    >>> result = verify_kkt(np.array([0.5, 0.5]), grad_f, h=h, grad_h=grad_h, eq_multipliers=np.array([-1.0]))
    >>> result.satisfied
    True
    """
    x = np.asarray(x, dtype=np.float64)
    stationarity = grad_f(x).astype(np.float64)
    primal_feasible = True
    dual_feasible = True
    complementary = True

    if h is not None:
        lam = np.zeros(np.atleast_1d(h(x)).shape[0]) if eq_multipliers is None else np.asarray(eq_multipliers, dtype=np.float64)
        stationarity = stationarity + grad_h(x).T @ lam
        primal_feasible = primal_feasible and bool(np.allclose(h(x), 0.0, atol=tol))

    if g is not None:
        mu = np.zeros(np.atleast_1d(g(x)).shape[0]) if ineq_multipliers is None else np.asarray(ineq_multipliers, dtype=np.float64)
        stationarity = stationarity + grad_g(x).T @ mu
        g_vals = np.atleast_1d(g(x))
        primal_feasible = primal_feasible and bool(np.all(g_vals <= tol))
        dual_feasible = bool(np.all(mu >= -tol))
        complementary = bool(np.allclose(mu * g_vals, 0.0, atol=tol))

    stat_residual = float(np.linalg.norm(stationarity))
    satisfied = (stat_residual < tol) and primal_feasible and dual_feasible and complementary
    return KKTResult(
        x=x,
        stationarity_residual=stat_residual,
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        complementary_slackness=complementary,
        satisfied=satisfied,
    )


class PenaltyMethod:
    r"""Quadratic penalty method: a sequence of unconstrained minimizations
    with an increasing penalty weight.

    Reformulates :math:`\min f(x)` s.t. :math:`h(x)=0`, :math:`g(x)\leq0`
    as the unconstrained problem :math:`\min_x f(x) + \mu \left[\sum_i
    h_i(x)^2 + \sum_j \max(0, g_j(x))^2\right]`, solved for a sequence of
    increasing :math:`\mu`, each warm-started from the previous
    solution -- as :math:`\mu \to \infty` the penalized minimizer
    converges to a KKT point of the original constrained problem. Each
    unconstrained sub-solve uses
    :class:`~mathkit.optimization.systems.newton_quasi_newton.BFGS` by
    default. See Nocedal & Wright, *Numerical Optimization*, 2nd ed.,
    Ch. 17.1 ("The Quadratic Penalty Method").

    Parameters
    ----------
    mu0 : float
        Initial penalty weight.
    mu_factor : float
        Multiplicative increase applied to ``mu`` after each sub-solve.
    n_outer : int
        Number of penalty-weight increases (outer iterations).
    inner_optimizer : UnconstrainedOptimizer, optional
        Solver used for each unconstrained sub-problem; defaults to
        :class:`~mathkit.optimization.systems.newton_quasi_newton.BFGS`.

    Examples
    --------
    >>> import numpy as np
    >>> # minimize x^2 + y^2 subject to x + y = 1: exact solution (0.5, 0.5).
    >>> f = lambda z: z[0] ** 2 + z[1] ** 2
    >>> grad_f = lambda z: np.array([2.0 * z[0], 2.0 * z[1]])
    >>> h = lambda z: np.array([z[0] + z[1] - 1.0])
    >>> grad_h = lambda z: np.array([[1.0, 1.0]])
    >>> result = PenaltyMethod(n_outer=8).minimize(f, grad_f, np.array([0.0, 0.0]), h=h, grad_h=grad_h)
    >>> np.allclose(result.x, [0.5, 0.5], atol=1e-3)
    True
    """

    def __init__(self, mu0: float = 1.0, mu_factor: float = 10.0, n_outer: int = 10, inner_optimizer: Optional[UnconstrainedOptimizer] = None):
        self.mu0 = float(mu0)
        self.mu_factor = float(mu_factor)
        self.n_outer = int(n_outer)
        self.inner_optimizer = inner_optimizer if inner_optimizer is not None else BFGS(tol=DEFAULT_RTOL, max_iter=DEFAULT_MAX_ITER)

    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        grad_h: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        grad_g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizeResult:
        """Run the sequential penalty minimization.

        Parameters
        ----------
        f, grad_f : callable
            Objective and its gradient.
        x0 : ndarray
        h, grad_h : callable, optional
            Equality constraints ``h(x) -> ndarray`` and their Jacobian
            ``grad_h(x) -> ndarray`` shape (m, n).
        g, grad_g : callable, optional
            Inequality constraints ``g(x) <= 0`` and their Jacobian.

        Returns
        -------
        OptimizeResult
            ``extra["mu_history"]`` records the penalty weight used at
            each outer iteration.
        """
        x = np.asarray(x0, dtype=np.float64).copy()
        mu = self.mu0
        path = [x.copy()]
        mu_history = []

        for _ in range(self.n_outer):

            def penalized_f(z, mu=mu):
                val = f(z)
                if h is not None:
                    val = val + mu * float(np.sum(np.atleast_1d(h(z)) ** 2))
                if g is not None:
                    val = val + mu * float(np.sum(np.maximum(0.0, np.atleast_1d(g(z))) ** 2))
                return val

            def penalized_grad(z, mu=mu):
                grad = grad_f(z).astype(np.float64).copy()
                if h is not None:
                    grad = grad + 2.0 * mu * grad_h(z).T @ np.atleast_1d(h(z))
                if g is not None:
                    active = np.maximum(0.0, np.atleast_1d(g(z)))
                    grad = grad + 2.0 * mu * grad_g(z).T @ active
                return grad

            sub_result = self.inner_optimizer.minimize(penalized_f, penalized_grad, x)
            x = sub_result.x
            path.extend(list(sub_result.path[1:]))
            mu_history.append(mu)
            mu *= self.mu_factor

        return OptimizeResult(
            x=x, fun=float(f(x)), path=np.array(path), iterations=len(path) - 1, converged=True, method="penalty", extra={"mu_history": np.array(mu_history)}
        )

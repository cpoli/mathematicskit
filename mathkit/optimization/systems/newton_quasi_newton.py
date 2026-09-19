r"""Newton's method and BFGS, via :func:`scipy.optimize.minimize`.

Both are second-order (or approximately second-order) methods where the
line search and Hessian/inverse-Hessian-approximation bookkeeping is
substantial, fiddly, and already correctly implemented in
``scipy.optimize`` -- mathkit does not reimplement it. The value-add
here is capturing the per-iterate path (via ``scipy.optimize.minimize``'s
``callback`` argument, which the raw scipy call doesn't expose on its
own) into the same :class:`~mathkit.optimization.core.base.OptimizeResult`
shape used by the hand-rolled methods in
:mod:`mathkit.optimization.systems.gradient_descent` and
:mod:`mathkit.optimization.systems.conjugate_gradient`, so all of them
can be compared on equal footing. See Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Ch. 6 (Newton) and Ch. 6.1 (BFGS).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import optimize as sopt

from mathkit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer

__all__ = ["NewtonMethod", "BFGS"]


class NewtonMethod(UnconstrainedOptimizer):
    r"""Newton's method via ``scipy.optimize.minimize(method="Newton-CG")``.

    Each step solves :math:`\nabla^2 f(x_k) p_k = -\nabla f(x_k)`
    (approximately, via a truncated conjugate-gradient sub-solve, which
    is what makes "Newton-CG" practical for large problems without
    forming/factoring the Hessian explicitly) and steps along a
    line-searched multiple of :math:`p_k`. Converges quadratically near a
    minimizer where the Hessian is positive-definite -- far faster than
    :mod:`~mathkit.optimization.systems.gradient_descent` on
    ill-conditioned problems like :func:`~mathkit.optimization.utils.test_functions.rosenbrock`.
    See Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 6 and
    Ch. 7.1 (Newton-CG).

    Parameters
    ----------
    tol : float
        Forwarded as ``scipy.optimize.minimize``'s ``tol``.
    max_iter : int
        Forwarded as ``options={"maxiter": max_iter}``.

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.optimization.utils.test_functions import rosenbrock, rosenbrock_grad, rosenbrock_hess
    >>> result = NewtonMethod().minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]), hess=rosenbrock_hess)
    >>> np.allclose(result.x, [1.0, 1.0], atol=1e-4)
    True
    """

    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        hess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizeResult:
        """Minimize ``f`` from ``x0``.

        Parameters
        ----------
        f, grad : callable
        x0 : ndarray
        hess : callable, optional
            ``hess(x) -> ndarray`` of shape ``(n, n)``. If omitted,
            scipy estimates Hessian-vector products from `grad` via
            finite differences.

        Returns
        -------
        OptimizeResult
        """
        x0 = np.asarray(x0, dtype=np.float64)
        path = [x0.copy()]

        def callback(xk):
            path.append(np.asarray(xk, dtype=np.float64).copy())

        res = sopt.minimize(f, x0, jac=grad, hess=hess, method="Newton-CG", tol=self.tol, callback=callback, options={"maxiter": self.max_iter})
        return OptimizeResult(
            x=res.x,
            fun=float(res.fun),
            path=np.array(path),
            iterations=int(res.nit),
            converged=bool(res.success),
            method="newton_cg",
            extra={"scipy_message": res.message},
        )


class BFGS(UnconstrainedOptimizer):
    r"""Quasi-Newton BFGS, via ``scipy.optimize.minimize(method="BFGS")``.

    Builds up an approximation to the inverse Hessian purely from
    successive gradient evaluations (the BFGS update), converging
    superlinearly without ever needing an explicit Hessian -- the
    standard practical alternative to Newton's method when second
    derivatives are unavailable or expensive. See Nocedal & Wright,
    *Numerical Optimization*, 2nd ed., Ch. 6.1, eq. (6.19).

    Parameters
    ----------
    tol : float
        Forwarded as ``scipy.optimize.minimize``'s ``tol``.
    max_iter : int
        Forwarded as ``options={"maxiter": max_iter}``.

    Examples
    --------
    >>> import numpy as np
    >>> from mathkit.optimization.utils.test_functions import rosenbrock, rosenbrock_grad
    >>> result = BFGS().minimize(rosenbrock, rosenbrock_grad, np.array([-1.2, 1.0]))
    >>> np.allclose(result.x, [1.0, 1.0], atol=1e-3)
    True
    """

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x0 = np.asarray(x0, dtype=np.float64)
        path = [x0.copy()]

        def callback(xk):
            path.append(np.asarray(xk, dtype=np.float64).copy())

        res = sopt.minimize(f, x0, jac=grad, method="BFGS", tol=self.tol, callback=callback, options={"maxiter": self.max_iter})
        return OptimizeResult(
            x=res.x,
            fun=float(res.fun),
            path=np.array(path),
            iterations=int(res.nit),
            converged=bool(res.success),
            method="bfgs",
            extra={"scipy_message": res.message},
        )

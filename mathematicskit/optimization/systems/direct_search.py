r"""Nelder-Mead simplex search, via ``scipy.optimize.minimize(method="Nelder-Mead")``.

John Nelder and Roger Mead (1965) minimized a function using only its
values: a simplex of :math:`n+1` points reflects, expands, contracts,
and shrinks across the landscape, adapting its shape to the local
contours. No gradient is needed, which made it the method of choice for
noisy or non-smooth experimental objectives. SciPy implements it
directly (with Gao and Han's adaptive parameters available), so it is
wrapped here only to record the iterate path. See J. A. Nelder and R.
Mead, "A Simplex Method for Function Minimization," The Computer Journal
7(4) (1965), 308-313.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import optimize as sopt

from mathematicskit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer

__all__ = ["NelderMead"]


class NelderMead(UnconstrainedOptimizer):
    r"""Derivative-free Nelder-Mead simplex search.

    Parameters
    ----------
    tol : float
        Forwarded as both ``xatol`` and ``fatol``.
    max_iter : int
        Forwarded as ``options={"maxiter": max_iter}``.

    Examples
    --------
    >>> import numpy as np
    >>> from mathematicskit.optimization.utils.test_functions import rosenbrock
    >>> result = NelderMead(tol=1e-10).minimize(rosenbrock, None, np.array([-1.2, 1.0]))
    >>> np.allclose(result.x, [1.0, 1.0], atol=1e-6)
    True
    """

    def minimize(self, f: Callable[[np.ndarray], float], grad: Optional[Callable[[np.ndarray], np.ndarray]], x0: np.ndarray) -> OptimizeResult:
        """Minimize ``f`` from ``x0``.

        Parameters
        ----------
        f : callable
        grad : callable or None
            Ignored -- Nelder-Mead uses function values only. Accepted so
            the method fits the shared
            :class:`~mathematicskit.optimization.core.base.UnconstrainedOptimizer`
            interface (e.g. :func:`~mathematicskit.optimization.utils.comparison.compare_optimizers`).
        x0 : ndarray

        Returns
        -------
        OptimizeResult
        """
        x0 = np.asarray(x0, dtype=np.float64)
        path = [x0.copy()]

        def callback(xk):
            path.append(np.asarray(xk, dtype=np.float64).copy())

        res = sopt.minimize(f, x0, method="Nelder-Mead", callback=callback, options={"maxiter": self.max_iter, "xatol": self.tol, "fatol": self.tol})
        return OptimizeResult(
            x=res.x,
            fun=float(res.fun),
            path=np.array(path),
            iterations=int(res.nit),
            converged=bool(res.success),
            method="nelder_mead",
            extra={"nfev": int(res.nfev), "scipy_message": res.message},
        )

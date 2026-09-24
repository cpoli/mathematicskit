r"""Nonlinear least squares by the Levenberg-Marquardt method, via
:func:`scipy.optimize.least_squares`.

To minimize :math:`\tfrac12 \|r(x)\|^2`, Levenberg (1944) and Marquardt
(1963) damp the Gauss-Newton step,

.. math::

   (J^T J + \lambda D)\, \delta = -J^T r,

where :math:`J` is the Jacobian of the residuals. Large :math:`\lambda`
gives a short gradient-descent step; small :math:`\lambda` gives the
fast Gauss-Newton step. SciPy's ``method="lm"`` wraps MINPACK's
trust-region implementation of this idea, so mathematicskit does not
reimplement it. See K. Levenberg, "A Method for the Solution of Certain
Non-Linear Problems in Least Squares," Quarterly of Applied Mathematics
2(2) (1944), 164-168; D. W. Marquardt, "An Algorithm for Least-Squares
Estimation of Nonlinear Parameters," Journal of the Society for
Industrial and Applied Mathematics 11(2) (1963), 431-441.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import optimize as sopt

from mathematicskit.optimization.core.base import LeastSquaresResult

__all__ = ["levenberg_marquardt"]


def levenberg_marquardt(
    residual: Callable[[np.ndarray], np.ndarray],
    x0,
    jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1e-10,
    max_nfev: Optional[int] = None,
) -> LeastSquaresResult:
    r"""Minimize :math:`\tfrac12 \sum_i r_i(x)^2` with the Levenberg-Marquardt method.

    Thin wrapper around ``scipy.optimize.least_squares(method="lm")``
    (MINPACK's ``lmder``/``lmdif``). Requires at least as many residuals
    as parameters.

    Parameters
    ----------
    residual : callable
        ``residual(x) -> ndarray`` of shape ``(m,)``.
    x0 : array_like, shape (n,)
        Initial parameter guess.
    jac : callable, optional
        ``jac(x) -> ndarray`` of shape ``(m, n)``; estimated by finite
        differences if omitted.
    tol : float
        Forwarded as ``ftol``, ``xtol`` and ``gtol``.
    max_nfev : int, optional
        Maximum number of residual evaluations.

    Returns
    -------
    LeastSquaresResult

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0.0, 4.0, 20)
    >>> y = 2.0 * np.exp(-0.5 * t)
    >>> result = levenberg_marquardt(lambda p: p[0] * np.exp(-p[1] * t) - y, [1.0, 1.0])
    >>> np.allclose(result.x, [2.0, 0.5])
    True
    """
    jac_arg = jac if jac is not None else "2-point"
    res = sopt.least_squares(residual, np.asarray(x0, dtype=np.float64), jac=jac_arg, method="lm", ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
    return LeastSquaresResult(x=res.x, cost=float(res.cost), residuals=res.fun, success=bool(res.success), nfev=int(res.nfev), message=res.message)

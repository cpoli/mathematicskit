r"""Momentum-based first-order methods: Nesterov's accelerated gradient and Adam.

Hand-rolled to expose the per-iterate path (neither is in
``scipy.optimize``).

- Y. E. Nesterov, "A Method of Solving a Convex Programming Problem with
  Convergence Rate :math:`O(1/k^2)`," Soviet Mathematics Doklady 27(2)
  (1983), 372-376.
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization,"
  3rd International Conference on Learning Representations (ICLR 2015),
  arXiv:1412.6980.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.optimization.core.base import OptimizeResult, UnconstrainedOptimizer

__all__ = ["NesterovAcceleratedGradient", "Adam"]


class NesterovAcceleratedGradient(UnconstrainedOptimizer):
    r"""Nesterov's accelerated gradient method (1983).

    Takes the gradient step from an extrapolated point:

    .. math::

       y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1}), \qquad
       x_{k+1} = y_k - \alpha \nabla f(y_k).

    For convex ``f`` with ``L``-Lipschitz gradient and
    :math:`\alpha = 1/L`, :math:`f(x_k) - f^* = O(1/k^2)`, against
    :math:`O(1/k)` for plain gradient descent -- the optimal rate for
    any first-order method.

    Parameters
    ----------
    alpha : float
        Step size (``1/L`` is the classical choice).
    tol : float
        Convergence tolerance on ``||grad(x)||``.
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: 0.5 * (x[0] ** 2 + 100.0 * x[1] ** 2)
    >>> grad = lambda x: np.array([x[0], 100.0 * x[1]])
    >>> result = NesterovAcceleratedGradient(alpha=0.01, tol=1e-8, max_iter=5000).minimize(f, grad, np.array([1.0, 1.0]))
    >>> np.allclose(result.x, [0.0, 0.0], atol=1e-6)
    True
    """

    def __init__(self, alpha: float = 0.1, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.alpha = float(alpha)

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64).copy()
        x_prev = x.copy()
        path = [x.copy()]
        converged = False
        it = 0
        while it < self.max_iter:
            if np.linalg.norm(grad(x)) < self.tol:
                converged = True
                break
            it += 1
            y = x + ((it - 1.0) / (it + 2.0)) * (x - x_prev)
            x_prev = x
            x = y - self.alpha * grad(y)
            path.append(x.copy())
        return OptimizeResult(x=x, fun=float(f(x)), path=np.array(path), iterations=it, converged=converged, method="nesterov")


class Adam(UnconstrainedOptimizer):
    r"""Adam: adaptive moment estimation (Kingma and Ba, 2014).

    Keeps exponential moving averages of the gradient and of its
    elementwise square, corrects their bias toward zero, and scales each
    coordinate's step by the root-mean-square gradient:

    .. math::

       m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k, \qquad
       v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2,

    .. math::

       x_{k+1} = x_k - \alpha \frac{m_k / (1-\beta_1^k)}{\sqrt{v_k/(1-\beta_2^k)} + \varepsilon}.

    Each coordinate moves by roughly ``alpha`` per step regardless of
    the gradient's scale, which makes Adam robust to badly scaled and
    noisy (stochastic) gradients. With a constant ``alpha`` it settles
    within about ``alpha`` of a minimizer rather than converging exactly.

    Parameters
    ----------
    alpha : float
        Step size.
    beta1, beta2 : float
        Decay rates of the first- and second-moment estimates.
    eps : float
        Denominator safeguard.
    tol : float
        Convergence tolerance on ``||grad(x)||``.
    max_iter : int

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda x: (x[0] - 1.0) ** 2 + 1000.0 * (x[1] + 2.0) ** 2
    >>> grad = lambda x: np.array([2.0 * (x[0] - 1.0), 2000.0 * (x[1] + 2.0)])
    >>> result = Adam(alpha=0.05, max_iter=3000).minimize(f, grad, np.array([0.0, 0.0]))
    >>> np.allclose(result.x, [1.0, -2.0], atol=1e-2)
    True
    """

    def __init__(
        self, alpha: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER
    ):
        super().__init__(tol=tol, max_iter=max_iter)
        self.alpha = float(alpha)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def minimize(self, f: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], x0: np.ndarray) -> OptimizeResult:
        x = np.asarray(x0, dtype=np.float64).copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        path = [x.copy()]
        converged = False
        it = 0
        while it < self.max_iter:
            g = np.asarray(grad(x), dtype=np.float64)
            if np.linalg.norm(g) < self.tol:
                converged = True
                break
            it += 1
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * g * g
            m_hat = m / (1.0 - self.beta1**it)
            v_hat = v / (1.0 - self.beta2**it)
            x = x - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
            path.append(x.copy())
        return OptimizeResult(x=x, fun=float(f(x)), path=np.array(path), iterations=it, converged=converged, method="adam")

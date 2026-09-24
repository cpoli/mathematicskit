r"""Stochastic approximation: the Robbins-Monro iteration.

Robbins and Monro (1951) showed how to find a root of a function
:math:`M(x)` that can only be observed through noise. Applied to a noisy
gradient :math:`Y_k = \nabla f(x_k) + \varepsilon_k`, the iteration

.. math::

   x_{k+1} = x_k - a_k Y_k, \qquad \sum_k a_k = \infty, \quad \sum_k a_k^2 < \infty,

is stochastic gradient descent. Hand-rolled: the iteration itself is the
subject, and SciPy has no equivalent. See H. Robbins and S. Monro, "A
Stochastic Approximation Method," The Annals of Mathematical Statistics
22(3) (1951), 400-407.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mathematicskit.optimization.core.base import OptimizeResult

__all__ = ["robbins_monro"]


def robbins_monro(
    noisy_grad: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    x0,
    a0: float = 1.0,
    n_iter: int = 1000,
    seed: Optional[int] = None,
    f: Optional[Callable[[np.ndarray], float]] = None,
) -> OptimizeResult:
    r"""Stochastic gradient descent with Robbins-Monro steps :math:`a_k = a_0/(k+1)`.

    Parameters
    ----------
    noisy_grad : callable
        ``noisy_grad(x, rng) -> ndarray``: an unbiased, noisy observation
        of :math:`\nabla f(x)` (or, more generally, of the function whose
        root is sought), drawing its randomness from ``rng``.
    x0 : array_like
        Starting point.
    a0 : float
        Step-size scale; the step at iteration ``k`` (from 0) is ``a0 / (k + 1)``.
    n_iter : int
        Number of iterations (there is no reliable stopping test on noisy data).
    seed : int, optional
        Seed for the :class:`numpy.random.Generator` passed to ``noisy_grad``.
    f : callable, optional
        The (noise-free) objective, used only to report ``fun``; ``nan`` if omitted.

    Returns
    -------
    OptimizeResult

    Notes
    -----
    For :math:`f(x) = \tfrac12\|x - \mu\|^2` and ``a0 = 1`` the iterate
    after ``n`` steps is exactly the sample mean of the ``n`` noisy
    observations of :math:`\mu` -- the running average is the simplest
    stochastic-approximation scheme.

    Examples
    --------
    >>> import numpy as np
    >>> noisy = lambda x, rng: x - 3.0 + rng.normal(size=x.shape)
    >>> result = robbins_monro(noisy, [0.0], n_iter=20000, seed=0)
    >>> bool(abs(result.x[0] - 3.0) < 0.05)
    True
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=np.float64).copy()
    path = [x.copy()]
    for k in range(int(n_iter)):
        x = x - (a0 / (k + 1)) * np.asarray(noisy_grad(x, rng), dtype=np.float64)
        path.append(x.copy())
    fun = float(f(x)) if f is not None else float("nan")
    return OptimizeResult(x=x, fun=fun, path=np.array(path), iterations=int(n_iter), converged=True, method="robbins_monro")

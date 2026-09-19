"""Error/stability analysis utilities shared across
mathkit.numerical_analysis's root-finding, interpolation, and regression
systems.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_convergence_order", "lebesgue_constant", "condition_number"]


def estimate_convergence_order(history: np.ndarray, root: float) -> float:
    r"""Estimate an iterative sequence's empirical order of convergence.

    Given the last four iterates :math:`x_{n-2}, x_{n-1}, x_n, x_{n+1}`
    and errors :math:`e_k = |x_k - \text{root}|`, the order :math:`p`
    satisfying :math:`e_{k+1} \approx C e_k^p` is estimated from

    .. math::

        p \approx \frac{\ln(e_{n+1} / e_n)}{\ln(e_n / e_{n-1})}

    using the two most recent error ratios. See Burden & Faires,
    *Numerical Analysis*, 10th ed., Ch. 2.4 ("Error Analysis for
    Iterative Methods"), Definition 2.8 and the surrounding discussion.

    Parameters
    ----------
    history : ndarray, shape (n,)
        Sequence of iterates, e.g. ``RootResult.history``; needs at
        least 4 entries.
    root : float
        The (numerically) exact root the sequence converges to, e.g.
        ``RootResult.root`` or a known closed-form value.

    Returns
    -------
    float
        The estimated order :math:`p` (``1.0`` for linear convergence,
        e.g. bisection/secant's theoretical rate is between 1 and the
        golden ratio ~1.618; ``2.0`` for Newton's quadratic convergence
        at a simple root).

    Examples
    --------
    >>> import numpy as np
    >>> # A synthetic quadratically-convergent sequence: e_{k+1} = e_k^2.
    >>> errors = [0.1, 0.01, 0.0001, 1e-8]
    >>> history = np.array([1.0 - e for e in errors])
    >>> round(estimate_convergence_order(history, 1.0), 2)
    2.0
    """
    history = np.asarray(history, dtype=np.float64)
    if history.shape[0] < 4:
        raise ValueError("need at least 4 iterates to estimate convergence order")
    errors = np.abs(history - root)
    # Fast (quadratic/superlinear) methods can reach float64 roundoff
    # within a handful of iterations; once the error is at that floor,
    # consecutive ratios are dominated by rounding noise rather than the
    # method's true asymptotic rate, so restrict to errors still well
    # above roundoff (a generous 1e-10 floor) before taking the most
    # recent three -- the last iterates *reliably* reflecting the
    # method's convergence order, not its behavior after it has already
    # converged to machine precision.
    reliable = np.flatnonzero(errors > 1e-10)
    if reliable.shape[0] < 3:
        reliable = np.flatnonzero(errors > 0.0)
    if reliable.shape[0] < 3:
        raise ValueError("need at least 3 iterates with nonzero error to estimate convergence order")
    i_nm1, i_n, i_np1 = reliable[-3:]
    e_nm1, e_n, e_np1 = errors[i_nm1], errors[i_n], errors[i_np1]
    return float(np.log(e_np1 / e_n) / np.log(e_n / e_nm1))


def lebesgue_constant(nodes: np.ndarray, n_eval: int = 2000) -> float:
    r"""Estimate the Lebesgue constant of a set of interpolation nodes.

    The Lebesgue constant :math:`\Lambda_n = \max_x \sum_i |L_i(x)|`
    (where :math:`L_i` are the Lagrange basis polynomials for `nodes`)
    bounds how much interpolation error can be amplified relative to the
    best possible polynomial approximation,
    :math:`\|f - p_n\|_\infty \leq (1 + \Lambda_n) \|f - p_n^*\|_\infty`,
    and so serves as the interpolation problem's condition number: large
    :math:`\Lambda_n` (as for equally spaced nodes, where it grows
    exponentially in `n`) signals the Runge-phenomenon-prone regime,
    while Chebyshev nodes keep it growing only like :math:`O(\log n)`.
    See Trefethen, *Approximation Theory and Approximation Practice*,
    2013, Ch. 15.

    Parameters
    ----------
    nodes : ndarray, shape (n + 1,)
        Interpolation nodes.
    n_eval : int
        Number of evaluation points spanning ``[min(nodes), max(nodes)]``
        used to approximate the max.

    Returns
    -------
    float
        The (approximate) Lebesgue constant.

    Examples
    --------
    >>> import numpy as np
    >>> equally_spaced = np.linspace(-1, 1, 15)
    >>> chebyshev = np.cos(np.pi * (2 * np.arange(15) + 1) / (2 * 15))
    >>> lebesgue_constant(equally_spaced) > lebesgue_constant(chebyshev)
    True
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    xs = np.linspace(nodes.min(), nodes.max(), n_eval)
    total = np.zeros(n_eval)
    for i in range(n):
        li = np.ones(n_eval)
        for j in range(n):
            if j == i:
                continue
            li *= (xs - nodes[j]) / (nodes[i] - nodes[j])
        total += np.abs(li)
    return float(np.max(total))


def condition_number(matrix: np.ndarray) -> float:
    r"""2-norm condition number :math:`\kappa_2(A) = \sigma_{\max}/\sigma_{\min}`, via :func:`numpy.linalg.cond`.

    Used to flag numerically unstable fits/interpolation problems, e.g.
    :func:`~mathkit.numerical_analysis.systems.regression.PolynomialRegression`'s
    Vandermonde design matrix, whose condition number grows quickly with
    polynomial degree. See Burden & Faires, *Numerical Analysis*, 10th
    ed., Ch. 7.5, and Trefethen & Bau, *Numerical Linear Algebra*, 1997,
    Lecture 12.

    Parameters
    ----------
    matrix : ndarray, shape (m, n)

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> A = np.diag([1.0, 2.0, 100.0])
    >>> round(condition_number(A), 4)
    100.0
    """
    cond = np.linalg.cond(np.asarray(matrix, dtype=np.float64), p=2)
    if not np.isfinite(cond):
        raise np.linalg.LinAlgError("matrix is singular (or the zero matrix)")
    return float(cond)

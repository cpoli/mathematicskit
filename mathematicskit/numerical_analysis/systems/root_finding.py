"""Scalar root-finding methods: bisection, Newton-Raphson, secant, and
fixed-point iteration.

All four are textbook methods; see Burden & Faires, *Numerical Analysis*,
10th ed., Ch. 2 ("Solutions of Equations in One Variable") for the
derivations and convergence-order proofs each docstring below references.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mathematicskit.constants import DEFAULT_MAX_ITER, DEFAULT_RTOL
from mathematicskit.numerical_analysis.core.base import IterativeRootFinder, RootResult

__all__ = ["Bisection", "NewtonRaphson", "Secant", "FixedPointIteration"]


class Bisection(IterativeRootFinder):
    """Bisection method on a bracket ``[a, b]`` with ``f(a) * f(b) < 0``.

    Halves the bracket each iteration, keeping the half where the sign
    change persists; converges linearly (order 1) with error bound
    :math:`|x_n - \\text{root}| \\leq (b - a) / 2^{n+1}`. See Burden &
    Faires, *Numerical Analysis*, 10th ed., Ch. 2.1.

    Parameters
    ----------
    f : callable
        Continuous function ``f(x) -> float``.
    a, b : float
        Bracket endpoints with ``f(a)`` and ``f(b)`` of opposite sign.
    tol : float
        Stop once the bracket width is below `tol`.
    max_iter : int
        Maximum number of bisections.

    Examples
    --------
    >>> result = Bisection(lambda x: x**2 - 2.0, 0.0, 2.0, tol=1e-10).solve()
    >>> round(result.root, 6)
    1.414214
    >>> result.converged
    True
    >>> # ``history`` holds exactly one midpoint per iteration performed.
    >>> len(result.history) == result.iterations
    True
    """

    def __init__(self, f: Callable[[float], float], a: float, b: float, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        fa, fb = f(a), f(b)
        if fa == 0.0:
            self._root_immediate: Optional[float] = a
        elif fb == 0.0:
            self._root_immediate = b
        else:
            if fa * fb > 0:
                raise ValueError("f(a) and f(b) must have opposite signs")
            self._root_immediate = None
        self.f = f
        self.a, self.b = float(a), float(b)

    def solve(self) -> RootResult:
        if self._root_immediate is not None:
            return RootResult(root=self._root_immediate, converged=True, iterations=0, history=np.array([self._root_immediate]), method="bisection")
        a, b = self.a, self.b
        fa = self.f(a)
        # The first recorded iterate is produced by the loop below; seeding
        # ``history`` with the same midpoint here would duplicate it.
        history: list = []
        converged = False
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            c = 0.5 * (a + b)
            fc = self.f(c)
            history.append(c)
            if fc == 0.0 or (b - a) / 2.0 < self.tol:  # noqa: SIM102
                converged = True
                break
            if fa * fc < 0:
                b = c
            else:
                a, fa = c, fc
        return RootResult(root=history[-1], converged=converged, iterations=n_iter, history=np.array(history), method="bisection", extra={"bracket": (a, b)})


class NewtonRaphson(IterativeRootFinder):
    """Newton-Raphson method: :math:`x_{n+1} = x_n - f(x_n) / f'(x_n)`.

    Converges quadratically (order 2) near a simple root, provided
    :math:`f'` doesn't vanish there. See Burden & Faires, *Numerical
    Analysis*, 10th ed., Ch. 2.3, Theorem 2.9.

    Parameters
    ----------
    f, fprime : callable
        Function and its derivative, both ``float -> float``.
    x0 : float
        Initial guess.
    tol : float
        Stop once ``|x_{n+1} - x_n| < tol``.
    max_iter : int
        Maximum number of iterations.

    Examples
    --------
    >>> result = NewtonRaphson(lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0=1.0).solve()
    >>> round(result.root, 10)
    1.4142135624
    """

    def __init__(self, f: Callable[[float], float], fprime: Callable[[float], float], x0: float, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.f = f
        self.fprime = fprime
        self.x0 = float(x0)

    def solve(self) -> RootResult:
        x = self.x0
        history = [x]
        converged = False
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            fpx = self.fprime(x)
            if fpx == 0.0:
                raise ZeroDivisionError(f"f'(x) vanished at x={x} before convergence")
            x_new = x - self.f(x) / fpx
            history.append(x_new)
            if abs(x_new - x) < self.tol:
                x = x_new
                converged = True
                break
            x = x_new
        return RootResult(root=x, converged=converged, iterations=n_iter, history=np.array(history), method="newton_raphson")


class Secant(IterativeRootFinder):
    """Secant method: Newton's method with the derivative replaced by a
    finite-difference slope through the two most recent iterates.

    :math:`x_{n+1} = x_n - f(x_n) \\dfrac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}`.
    Converges superlinearly with order :math:`(1+\\sqrt5)/2 \\approx 1.618`
    (the golden ratio) near a simple root -- see Burden & Faires,
    *Numerical Analysis*, 10th ed., Ch. 2.3, Theorem 2.10 -- without
    requiring an analytic derivative.

    Parameters
    ----------
    f : callable
        Function ``f(x) -> float``.
    x0, x1 : float
        Two initial guesses.
    tol : float
        Stop once ``|x_{n+1} - x_n| < tol``.
    max_iter : int
        Maximum number of iterations.

    Examples
    --------
    >>> result = Secant(lambda x: x**2 - 2.0, x0=1.0, x1=2.0).solve()
    >>> round(result.root, 8)
    1.41421356
    """

    def __init__(self, f: Callable[[float], float], x0: float, x1: float, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.f = f
        self.x0, self.x1 = float(x0), float(x1)

    def solve(self) -> RootResult:
        x_prev, x_curr = self.x0, self.x1
        f_prev = self.f(x_prev)
        history = [x_prev, x_curr]
        converged = False
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            f_curr = self.f(x_curr)
            denom = f_curr - f_prev
            if denom == 0.0:
                raise ZeroDivisionError("f(x_n) - f(x_{n-1}) vanished before convergence")
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            history.append(x_new)
            if abs(x_new - x_curr) < self.tol:
                x_curr = x_new
                converged = True
                break
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
        return RootResult(root=x_curr, converged=converged, iterations=n_iter, history=np.array(history), method="secant")


class FixedPointIteration(IterativeRootFinder):
    """Fixed-point iteration: :math:`x_{n+1} = g(x_n)`, converging to a
    fixed point ``x* = g(x*)`` (equivalently a root of ``f(x) = g(x) - x``).

    Converges linearly if ``|g'(x*)| < 1`` in a neighborhood of the fixed
    point (the contraction condition); see Burden & Faires, *Numerical
    Analysis*, 10th ed., Ch. 2.2, Theorem 2.4 (Fixed-Point Theorem) and
    Corollary 2.5.

    Parameters
    ----------
    g : callable
        Iteration function ``g(x) -> float``.
    x0 : float
        Initial guess.
    tol : float
        Stop once ``|x_{n+1} - x_n| < tol``.
    max_iter : int
        Maximum number of iterations.

    Examples
    --------
    >>> # x* solves x = cos(x); g(x) = cos(x) is a contraction near x*.
    >>> import numpy as np
    >>> result = FixedPointIteration(np.cos, x0=0.5, tol=1e-10).solve()
    >>> bool(abs(result.root - np.cos(result.root)) < 1e-9)
    True
    """

    def __init__(self, g: Callable[[float], float], x0: float, tol: float = DEFAULT_RTOL, max_iter: int = DEFAULT_MAX_ITER):
        super().__init__(tol=tol, max_iter=max_iter)
        self.g = g
        self.x0 = float(x0)

    def solve(self) -> RootResult:
        x = self.x0
        history = [x]
        converged = False
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            x_new = self.g(x)
            history.append(x_new)
            if abs(x_new - x) < self.tol:
                x = x_new
                converged = True
                break
            x = x_new
        return RootResult(root=x, converged=converged, iterations=n_iter, history=np.array(history), method="fixed_point")

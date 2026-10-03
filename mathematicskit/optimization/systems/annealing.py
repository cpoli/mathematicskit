r"""Simulated annealing.

Kirkpatrick, Gelatt and Vecchi (1983) borrowed the Metropolis algorithm
of statistical physics for optimization: treat the objective :math:`f`
as an energy, propose a random move, always accept a downhill move and
accept an uphill one of size :math:`\Delta f` with probability
:math:`e^{-\Delta f/T}`. At high temperature :math:`T` the walk roams
freely across barriers; as :math:`T` is lowered slowly it settles into
deep minima, as a slowly cooled metal settles into a low-energy crystal.
Hand-rolled, since the temperature schedule is the subject;
:func:`scipy.optimize.dual_annealing` is a production variant to check
against. See S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi,
"Optimization by Simulated Annealing," Science 220(4598) (1983),
671-680, and V. Černý, "Thermodynamical Approach to the Traveling
Salesman Problem," Journal of Optimization Theory and Applications 45
(1985), 41-51.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from mathematicskit.optimization.core.base import OptimizeResult

__all__ = ["simulated_annealing"]


def simulated_annealing(
    f: Callable[[Any], float],
    x0,
    neighbor: Optional[Callable[[Any, np.random.Generator], Any]] = None,
    step_size: float = 1.0,
    t0: float = 1.0,
    cooling: float | Callable[[int], float] = 0.999,
    n_iter: int = 10000,
    seed: int = 0,
) -> OptimizeResult:
    r"""Minimize ``f`` by simulated annealing.

    At step :math:`k` a candidate :math:`y` is drawn from the
    neighbourhood of the current state :math:`x` and accepted with the
    Metropolis probability :math:`\min(1, e^{-(f(y) - f(x))/T_k})`.

    Parameters
    ----------
    f : callable
        Objective (energy) ``f(x) -> float``.
    x0 : array_like or any
        Starting state. With the default ``neighbor`` it must be a real
        vector; with a custom ``neighbor`` it can be any object (a tour,
        a spin configuration, ...).
    neighbor : callable, optional
        ``neighbor(x, rng) -> y``, a random candidate near ``x``. Defaults
        to a Gaussian step of standard deviation ``step_size``.
    step_size : float
        Scale of the default Gaussian move.
    t0 : float
        Initial temperature, comparable to typical uphill moves of ``f``.
    cooling : float or callable
        A float :math:`\alpha < 1` gives the geometric schedule
        :math:`T_k = t_0\alpha^k` of Kirkpatrick et al.; a callable
        ``cooling(k) -> T_k`` gives any other schedule (Geman and Geman's
        logarithmic :math:`T_k = c/\log(k + 2)` guarantees convergence to
        a global minimum, but impractically slowly).
    n_iter : int
    seed : int

    Returns
    -------
    OptimizeResult
        ``x``/``fun`` are the best state found. For a vector state,
        ``path`` holds the current state at every step (shape
        ``(n_iter + 1, n)``); otherwise it is empty. ``extra`` holds
        ``"temperatures"``, ``"energies"`` (of the current state) and
        ``"acceptance_rate"``.

    Examples
    --------
    >>> import numpy as np
    >>> rastrigin = lambda x: 10 * x.size + float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
    >>> result = simulated_annealing(rastrigin, [4.3, -3.7], step_size=0.5, t0=10.0, cooling=0.999, n_iter=20000, seed=1)
    >>> bool(result.fun < 1e-2), np.round(result.x, 1) + 0.0
    (True, array([0., 0.]))
    """
    rng = np.random.default_rng(seed)
    vector_state = neighbor is None
    x: Any = np.atleast_1d(np.asarray(x0, dtype=np.float64)).copy() if vector_state else x0

    def gaussian_step(state, generator):
        return state + step_size * generator.standard_normal(state.shape)

    move = gaussian_step if neighbor is None else neighbor
    schedule = cooling if callable(cooling) else (lambda k: t0 * cooling**k)
    fx = float(f(x))
    best_x, best_f = x, fx
    path = [np.copy(x)] if vector_state else []
    temperatures = np.empty(int(n_iter))
    energies = np.empty(int(n_iter) + 1)
    energies[0] = fx
    accepted = 0
    for k in range(int(n_iter)):
        temperature = float(schedule(k))
        temperatures[k] = temperature
        y = move(x, rng)
        fy = float(f(y))
        delta = fy - fx
        if delta <= 0 or (temperature > 0 and rng.uniform() < np.exp(-delta / temperature)):
            x, fx = y, fy
            accepted += 1
            if fx < best_f:
                best_x, best_f = x, fx
        energies[k + 1] = fx
        if vector_state:
            path.append(np.copy(x))
    return OptimizeResult(
        x=best_x,
        fun=best_f,
        path=np.array(path),
        iterations=int(n_iter),
        converged=True,
        method="simulated_annealing",
        extra={"temperatures": temperatures, "energies": energies, "acceptance_rate": accepted / max(int(n_iter), 1)},
    )

r"""Lyapunov exponent estimation for 1D maps and simple flows.

See Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.5
("Lyapunov Exponent"), and Wolf et al. (1985), *Determining Lyapunov
exponents from a time series*, Physica D 16, for the flow variant.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["lyapunov_exponent_1d_map", "lyapunov_exponent_flow"]


def lyapunov_exponent_1d_map(
    f: Callable[[float], float], fprime: Callable[[float], float], x0: float, n_transient: int = 500, n_iterations: int = 2000
) -> float:
    r"""Lyapunov exponent of a 1D map, :math:`\lambda = \lim_{n\to\infty} \tfrac1n \sum_{k=0}^{n-1} \ln|f'(x_k)|`.

    Measures the average exponential rate of separation of nearby
    orbits: :math:`\lambda > 0` signals sensitive dependence on initial
    conditions (chaos), :math:`\lambda < 0` a stable periodic orbit or
    fixed point, :math:`\lambda = 0` marks a bifurcation point. See
    Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 10.5, eq.
    (10.5.1)-(10.5.2).

    Parameters
    ----------
    f, fprime : callable
        The map and its derivative.
    x0 : float
        Initial condition.
    n_transient : int
        Iterations discarded before averaging.
    n_iterations : int
        Iterations averaged over.

    Returns
    -------
    float

    Examples
    --------
    >>> # Logistic map at r=4 is exactly conjugate to a tent map with
    >>> # known Lyapunov exponent ln(2) ~ 0.693.
    >>> f = lambda x: 4.0 * x * (1.0 - x)
    >>> fprime = lambda x: 4.0 - 8.0 * x
    >>> lam = lyapunov_exponent_1d_map(f, fprime, x0=0.4, n_iterations=100000)
    >>> round(float(lam), 2)
    0.69
    """
    x = float(x0)
    for _ in range(n_transient):
        x = f(x)
    total = 0.0
    count = 0
    for _ in range(n_iterations):
        deriv = abs(fprime(x))
        if deriv > 1e-300:
            total += np.log(deriv)
            count += 1
        x = f(x)
    return total / count if count else float("nan")


def lyapunov_exponent_flow(rhs: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, dt: float = 0.01, n_steps: int = 20000, renorm_every: int = 10) -> float:
    r"""Largest Lyapunov exponent of a flow, via a shadow-trajectory method.

    Integrates the system alongside a nearby "shadow" trajectory,
    periodically measuring and renormalizing their separation (Benettin
    et al.'s algorithm) -- this avoids the naive approach of just
    watching two nearby trajectories diverge, which overflows once they
    separate enough to leave the linear regime. Uses simple forward-Euler
    stepping (adequate for an exponent estimate, where the log-average
    washes out an O(dt) local error) rather than
    :mod:`mathematicskit.integrators`, since renormalization must happen *between*
    steps at a resolution finer than any fixed-step RK4/Dormand-Prince
    call would expose. See Wolf et al. (1985), Physica D 16, and
    Strogatz, *Nonlinear Dynamics and Chaos*, 2nd ed., Ch. 9.3.

    Parameters
    ----------
    rhs : callable
        ``rhs(state) -> dstate/dt`` (autonomous).
    x0 : ndarray
        Initial condition.
    dt : float
        Integration step.
    n_steps : int
        Total steps.
    renorm_every : int
        Steps between renormalizations of the shadow separation.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> # A simple contracting linear flow has a negative exponent.
    >>> rhs = lambda x: -2.0 * x
    >>> lam = lyapunov_exponent_flow(rhs, np.array([1.0]), n_steps=5000)
    >>> bool(lam < 0)
    True
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    eps = 1e-8
    shadow = x + eps * np.ones_like(x) / np.linalg.norm(np.ones_like(x))
    log_sum = 0.0
    n_renorms = 0
    for step in range(1, n_steps + 1):
        x = x + dt * rhs(x)
        shadow = shadow + dt * rhs(shadow)
        if step % renorm_every == 0:
            sep = shadow - x
            dist = np.linalg.norm(sep)
            if dist > 1e-300:
                log_sum += np.log(dist / eps)
                n_renorms += 1
                shadow = x + (sep / dist) * eps
    total_time = n_renorms * renorm_every * dt
    return log_sum / total_time if total_time > 0 else float("nan")

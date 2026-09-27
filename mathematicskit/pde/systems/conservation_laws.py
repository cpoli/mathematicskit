r"""Inviscid Burgers' equation :math:`u_t + (u^2/2)_x = 0`: shocks and Godunov's method.

Nonlinear conservation laws steepen smooth data into shocks, where the
PDE no longer holds pointwise and only the integral (conservation) form
makes sense. Conservative finite-volume schemes update cell averages by
flux differences,

.. math:: u_j^{n+1} = u_j^n - \frac{\Delta t}{\Delta x}\left(F_{j+1/2} - F_{j-1/2}\right),

so shocks move at the Rankine-Hugoniot speed automatically. Godunov
(1959) took :math:`F_{j+1/2}` from the *exact* solution of the Riemann
problem between neighboring cells; for Burgers' convex flux that is

.. math::

    F = \begin{cases} \min_{u_L \le u \le u_R} u^2/2 & u_L \le u_R \\
                      \max(u_L^2/2,\ u_R^2/2) & u_L > u_R \end{cases}

The scheme is monotone and first order, and Godunov's theorem explains
why: a *linear* monotone (non-oscillatory) scheme cannot be better than
first order. Lax-Wendroff, second order, oscillates at shocks. Time
stepping is hand-rolled (scipy has no finite-volume solvers). See
LeVeque, *Finite Volume Methods for Hyperbolic Problems*, CUP 2002,
Ch. 11-12, and Godunov, "A difference method for numerical calculation
of discontinuous solutions of the equations of hydrodynamics," *Mat.
Sbornik* 47 (1959), 271-306.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.pde.core.base import PDESolution, _step_count, _thinned_indices
from mathematicskit.pde.systems.heat import InitialCondition, _sample
from mathematicskit.pde.systems.stability import check_cfl

__all__ = ["BurgersConservationLaw1D", "burgers_riemann_solution", "total_variation"]

_SCHEMES = ("godunov", "lax_friedrichs", "lax_wendroff")


def _flux(u):
    return 0.5 * u**2


def _godunov_flux(uL, uR):
    rarefaction = np.where(uL > 0.0, _flux(uL), np.where(uR < 0.0, _flux(uR), 0.0))
    shock = np.maximum(_flux(uL), _flux(uR))
    return np.where(uL <= uR, rarefaction, shock)


def total_variation(u) -> float:
    r"""Total variation :math:`\sum_j |u_{j+1} - u_j|` of grid values.

    Monotone schemes such as Godunov's never increase it (they are TVD);
    oscillatory ones such as Lax-Wendroff do at shocks.

    Parameters
    ----------
    u : array-like

    Returns
    -------
    float

    Examples
    --------
    >>> total_variation([0.0, 1.0, 0.0, 2.0])
    4.0
    """
    return float(np.sum(np.abs(np.diff(np.asarray(u, dtype=np.float64)))))


def burgers_riemann_solution(u_left: float, u_right: float, x, t: float, x0: float = 0.0) -> np.ndarray:
    r"""Exact entropy solution of the Burgers Riemann problem with a jump at `x0`.

    A shock moving at the Rankine-Hugoniot speed :math:`s = (u_L + u_R)/2`
    if :math:`u_L > u_R`; otherwise a rarefaction fan
    :math:`u = (x - x_0)/t` between :math:`u_L` and :math:`u_R`.

    Parameters
    ----------
    u_left, u_right : float
    x : array-like
    t : float
        Positive time.
    x0 : float

    Returns
    -------
    ndarray

    Examples
    --------
    >>> burgers_riemann_solution(1.0, 0.0, [0.4, 0.6], t=1.0).tolist()  # shock at x = 0.5
    [1.0, 0.0]
    >>> burgers_riemann_solution(0.0, 1.0, [-0.5, 0.25, 2.0], t=1.0).tolist()  # rarefaction
    [0.0, 0.25, 1.0]
    """
    xi = (np.asarray(x, dtype=np.float64) - x0) / t
    if u_left > u_right:
        return np.where(xi < 0.5 * (u_left + u_right), u_left, u_right)
    return np.clip(xi, u_left, u_right)


class BurgersConservationLaw1D:
    r"""Inviscid Burgers' equation :math:`u_t + (u^2/2)_x = 0` on :math:`[a, b]`, finite-volume schemes.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
        Initial cell values, sampled at the cell centers.
    x_range : tuple of float
    n : int
        Number of cells.
    bc : {"outflow", "periodic"}
        ``"outflow"`` copies the edge cells into ghost cells (waves leave freely).

    Examples
    --------
    >>> import numpy as np
    >>> law = BurgersConservationLaw1D(lambda x: np.where(x < 0.0, 1.0, 0.0), x_range=(-1.0, 1.0), n=200)
    >>> sol = law.solve(0.8, dt=0.005, scheme="godunov")
    >>> exact = burgers_riemann_solution(1.0, 0.0, sol.x, 0.8)
    >>> bool(np.mean(np.abs(sol.final - exact)) < 0.01)
    True
    """

    def __init__(self, u0: InitialCondition, x_range=(0.0, 1.0), n: int = 200, bc: str = "outflow"):
        if bc not in ("outflow", "periodic"):
            raise ValueError(f"Unknown boundary condition '{bc}'")
        a, b = float(x_range[0]), float(x_range[1])
        self.dx = (b - a) / n
        self.x = a + self.dx * (np.arange(n) + 0.5)
        self.bc = bc
        self.u0 = _sample(u0, self.x)

    def _with_ghosts(self, u: np.ndarray) -> np.ndarray:
        if self.bc == "periodic":
            return np.concatenate([[u[-1]], u, [u[0]]])
        return np.concatenate([[u[0]], u, [u[-1]]])

    def step(self, u: np.ndarray, dt: float, scheme: str = "godunov") -> np.ndarray:
        """Advance cell values `u` by one conservative step of `scheme`.

        Parameters
        ----------
        u : ndarray, shape (n,)
        dt : float
        scheme : {"godunov", "lax_friedrichs", "lax_wendroff"}

        Returns
        -------
        ndarray, shape (n,)
        """
        g = self._with_ghosts(u)
        uL, uR = g[:-1], g[1:]  # the two states at each of the n + 1 interfaces
        lam = dt / self.dx
        if scheme == "godunov":
            F = _godunov_flux(uL, uR)
        elif scheme == "lax_friedrichs":
            F = 0.5 * (_flux(uL) + _flux(uR)) - 0.5 / lam * (uR - uL)
        elif scheme == "lax_wendroff":
            speed = 0.5 * (uL + uR)
            F = 0.5 * (_flux(uL) + _flux(uR)) - 0.5 * lam * speed * (_flux(uR) - _flux(uL))
        else:
            raise ValueError(f"Unknown scheme '{scheme}'; expected one of {_SCHEMES}")
        return u - lam * (F[1:] - F[:-1])

    def solve(self, t_final: float, dt: float, scheme: str = "godunov", save_every: int = 1) -> PDESolution:
        """Step from ``t = 0`` to `t_final`.

        Parameters
        ----------
        t_final, dt : float
            `dt` is shrunk if needed so a whole number of steps lands exactly on `t_final`.
        scheme : {"godunov", "lax_friedrichs", "lax_wendroff"}
        save_every : int

        Returns
        -------
        PDESolution
            ``extra["stability"]`` is the CFL check with speed ``max |u0|``.
        """
        if scheme not in _SCHEMES:
            raise ValueError(f"Unknown scheme '{scheme}'; expected one of {_SCHEMES}")
        n_steps, dt = _step_count(t_final, dt)
        keep = _thinned_indices(n_steps + 1, save_every)
        saved = np.empty((len(keep), len(self.x)))
        u = self.u0.copy()
        slot = 0
        for k in range(n_steps + 1):
            if slot < len(keep) and keep[slot] == k:
                saved[slot] = u
                slot += 1
            if k < n_steps:
                u = self.step(u, dt, scheme)
        speed = float(np.max(np.abs(self.u0)))
        return PDESolution(t=keep * dt, x=self.x, u=saved, method=scheme, extra={"stability": check_cfl(speed, dt, self.dx, scheme)})

r"""Linear advection :math:`u_t + c\,u_x = 0` on a periodic domain, stepped by classic explicit schemes.

The four schemes differ only in how they difference :math:`u_x` and how
much numerical diffusion they add; each one's stability is governed by
the Courant number :math:`\nu = |c|\,\Delta t/\Delta x`
(Courant, Friedrichs & Lewy, 1928):

======================  ======================================  ==================
scheme                  update                                  stable for
======================  ======================================  ==================
``"upwind"``            one-sided difference against the flow   :math:`\nu \le 1`
``"lax_friedrichs"``    centered difference, neighbor average   :math:`\nu \le 1`
``"lax_wendroff"``      second order (Lax & Wendroff, 1960)     :math:`\nu \le 1`
``"ftcs"``              forward time, centered space            never
======================  ======================================  ==================

The time stepping is hand-rolled: the per-step update *is* the
pedagogical content here (scipy has no finite-difference PDE schemes).
See LeVeque, *Finite Difference Methods for Ordinary and Partial
Differential Equations*, SIAM 2007, Ch. 10.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.pde.core.base import PDESolution, _step_count, _thinned_indices
from mathematicskit.pde.systems.heat import InitialCondition, _sample
from mathematicskit.pde.systems.stability import check_cfl
from mathematicskit.pde.utils.operators import uniform_grid

__all__ = ["AdvectionEquation1D"]

_SCHEMES = ("upwind", "lax_friedrichs", "lax_wendroff", "ftcs")


class AdvectionEquation1D:
    r"""Linear advection :math:`u_t + c\,u_x = 0` on the periodic interval :math:`[0, L)`.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
        Initial profile.
    length : float
    n : int
        Grid points (the right endpoint, equal to the left one, is omitted).
    c : float
        Advection speed (either sign).

    Examples
    --------
    >>> import numpy as np
    >>> adv = AdvectionEquation1D(lambda x: np.sin(2 * np.pi * x), n=200)
    >>> sol = adv.solve(1.0, dt=0.004, scheme="lax_wendroff")  # one full period
    >>> sol.extra["stability"].number
    0.8
    >>> bool(np.max(np.abs(sol.final - adv.exact(1.0))) < 1e-2)
    True
    """

    def __init__(self, u0: InitialCondition, length: float = 1.0, n: int = 200, c: float = 1.0):
        self.length, self.c = float(length), float(c)
        self.x, self.dx = uniform_grid(0.0, self.length, n, periodic=True)
        self._u0_callable = u0 if callable(u0) else None
        self.u0 = _sample(u0, self.x)

    def exact(self, t: float) -> np.ndarray:
        """The exact solution ``u0(x - c t)`` (periodically wrapped) on the grid.

        Parameters
        ----------
        t : float

        Returns
        -------
        ndarray, shape (n,)
        """
        shifted = np.mod(self.x - self.c * t, self.length)
        if self._u0_callable is not None:
            return np.asarray(self._u0_callable(shifted), dtype=np.float64)
        return np.interp(shifted, self.x, self.u0, period=self.length)

    def step(self, u: np.ndarray, dt: float, scheme: str = "upwind") -> np.ndarray:
        """Advance grid values `u` by one step of `scheme`.

        Parameters
        ----------
        u : ndarray, shape (n,)
        dt : float
        scheme : {"upwind", "lax_friedrichs", "lax_wendroff", "ftcs"}

        Returns
        -------
        ndarray, shape (n,)
        """
        sigma = self.c * dt / self.dx
        up, um = np.roll(u, -1), np.roll(u, 1)  # u_{j+1}, u_{j-1}
        if scheme == "upwind":
            return u - sigma * (u - um) if self.c >= 0.0 else u - sigma * (up - u)
        if scheme == "lax_friedrichs":
            return 0.5 * (up + um) - 0.5 * sigma * (up - um)
        if scheme == "lax_wendroff":
            return u - 0.5 * sigma * (up - um) + 0.5 * sigma**2 * (up - 2.0 * u + um)
        if scheme == "ftcs":
            return u - 0.5 * sigma * (up - um)
        raise ValueError(f"Unknown scheme '{scheme}'; expected one of {_SCHEMES}")

    def solve(self, t_final: float, dt: float, scheme: str = "upwind", save_every: int = 1) -> PDESolution:
        """Step from ``t = 0`` to `t_final`.

        Parameters
        ----------
        t_final, dt : float
            `dt` is shrunk if needed so a whole number of steps lands exactly on `t_final`.
        scheme : {"upwind", "lax_friedrichs", "lax_wendroff", "ftcs"}
        save_every : int
            Keep every `save_every`-th step (the last one is always kept).

        Returns
        -------
        PDESolution
            ``extra["stability"]`` is
            :func:`~mathematicskit.pde.systems.stability.check_cfl` for this step.
        """
        if scheme not in _SCHEMES:
            raise ValueError(f"Unknown scheme '{scheme}'; expected one of {_SCHEMES}")
        n_steps, dt = _step_count(t_final, dt)
        keep = _thinned_indices(n_steps + 1, save_every)
        u_saved = np.empty((len(keep), len(self.x)))
        u = self.u0.copy()
        slot = 0
        for k in range(n_steps + 1):
            if slot < len(keep) and keep[slot] == k:
                u_saved[slot] = u
                slot += 1
            if k < n_steps:
                u = self.step(u, dt, scheme)
        return PDESolution(t=keep * dt, x=self.x, u=u_saved, method=scheme, extra={"stability": check_cfl(self.c, dt, self.dx, scheme)})

"""Abstract base classes and result containers for mathematicskit.pde.

Time-dependent problems solved by the *method of lines* -- discretize
space, leaving a large ODE system ``dU/dt = F(U, t)`` in time -- share the
:class:`MethodOfLinesPDE` ABC, which hands that ODE system to the shared
:mod:`mathematicskit.integrators` exactly as
:class:`mathematicskit.ode_dynamics.core.base.FlowSystem` does: each
concrete class sets ``self._rhs_njit`` (a module-level ``@njit`` function
with the :data:`~mathematicskit.integrators.RHSFunc` signature
``rhs(state, t, params)``) and ``self.params`` (a ``float64`` array
carrying the grid spacing, coefficients, and boundary data), so a single
compiled right-hand side is reused across every grid size and parameter
value without recompiling.

Every solver returns a small dataclass (mirroring physicskit's
``SimulationResult`` pattern): :class:`PDESolution` for time-dependent
problems, :class:`EllipticSolution` for steady (Laplace/Poisson/boundary
value) problems, and :class:`StabilityResult` for CFL/diffusion-number
checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mathematicskit.integrators import dopri5_integrate, rk4_integrate

__all__ = ["PDESolution", "EllipticSolution", "StabilityResult", "MethodOfLinesPDE"]


@dataclass
class PDESolution:
    """Output of a time-dependent PDE solve, sampled on a fixed spatial grid.

    For a 1D problem ``u[k, i]`` is the solution at time ``t[k]`` and
    position ``x[i]``; for a 2D problem ``u[k, i, j]`` is the solution at
    ``(x[i], y[j])`` (``"ij"`` indexing). Boundary points are included.
    """

    t: np.ndarray
    """ndarray, shape (n_t,): Saved time samples."""

    x: np.ndarray
    """ndarray, shape (n_x,): Spatial grid along x."""

    u: np.ndarray
    """ndarray, shape (n_t, n_x) or (n_t, n_x, n_y): Solution at each saved time."""

    y: Optional[np.ndarray] = None
    """ndarray, shape (n_y,), optional: Spatial grid along y (2D problems only)."""

    method: str = ""
    """str: Time-stepping scheme used, e.g. ``"rk4"``, ``"dopri5"``,
    ``"leapfrog"``, ``"crank_nicolson"``, or ``"upwind"``."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics, e.g. ``"stability"`` (a
    :class:`StabilityResult` for the step size used) or ``"v"`` (the
    velocity field of a wave-equation solve)."""

    @property
    def final(self) -> np.ndarray:
        """ndarray: The solution at the last saved time, ``u[-1]``."""
        return self.u[-1]

    def snapshot(self, t: float) -> np.ndarray:
        """Return the saved solution at the time sample nearest to `t`.

        Parameters
        ----------
        t : float

        Returns
        -------
        ndarray, shape (n_x,) or (n_x, n_y)
        """
        return self.u[int(np.argmin(np.abs(self.t - t)))]


@dataclass
class EllipticSolution:
    """Output of a steady boundary-value solve (Laplace, Poisson, ``u'' = f``).

    ``u[i]`` (1D) or ``u[i, j]`` (2D, ``"ij"`` indexing) is the discrete
    solution at ``x[i]`` (and ``y[j]``), boundary points included.
    """

    x: np.ndarray
    """ndarray, shape (n_x,): Spatial grid along x."""

    u: np.ndarray
    """ndarray, shape (n_x,) or (n_x, n_y): Discrete solution."""

    y: Optional[np.ndarray] = None
    """ndarray, shape (n_y,), optional: Spatial grid along y (2D problems only)."""

    method: str = ""
    """str: ``"direct"``, ``"cg"``, ``"jacobi"``, ``"gauss_seidel"``,
    ``"sor"``, or ``"chebyshev"``."""

    residual_norm: float = 0.0
    """float: 2-norm of the residual of the assembled linear system at the solution."""

    solver_result: Optional[object] = None
    """IterativeSolveResult, optional: The full
    :class:`~mathematicskit.linalg.core.base.IterativeSolveResult` (residual
    history, iteration count) when an iterative :mod:`mathematicskit.linalg`
    solver was used."""


@dataclass
class StabilityResult:
    """Outcome of a CFL / diffusion-number stability check for one step size."""

    number: float
    """float: The dimensionless number checked -- the Courant number
    ``|c| dt / dx`` or the diffusion number ``alpha dt / dx^2``."""

    limit: float
    """float: The largest stable value of `number` for this scheme
    (``inf`` for an unconditionally stable scheme, ``0`` for an
    unconditionally unstable one)."""

    stable: bool
    """bool: Whether ``number <= limit``."""

    scheme: str = ""
    """str: The scheme the limit applies to."""

    quantity: str = ""
    """str: ``"courant"`` or ``"diffusion"``."""


class MethodOfLinesPDE(ABC):
    """Common base for a time-dependent PDE semi-discretized in space.

    Concrete subclasses set ``self._rhs_njit`` (an ``@njit`` function with
    signature ``(state, t, params) -> dstate``), ``self.params``, and
    ``self.state0`` (the flattened initial ODE state), and implement
    :meth:`_to_field` to turn a batch of ODE states back into grid values
    (re-attaching boundary points, reshaping 2D grids).
    """

    _rhs_njit = None
    params: np.ndarray = np.empty(0)
    state0: np.ndarray = np.empty(0)
    x: np.ndarray = np.empty(0)
    y: Optional[np.ndarray] = None

    @abstractmethod
    def _to_field(self, states: np.ndarray) -> np.ndarray:
        """Map ODE states, shape (n_t, dim), to grid values, shape (n_t, n_x[, n_y])."""

    def max_stable_dt(self, method: str = "rk4") -> float:
        """Largest step size for which `method` is linearly stable on this grid.

        Subclasses override this; the default ``inf`` means "no known limit".
        """
        return np.inf

    def stability(self, dt: float, method: str = "rk4") -> StabilityResult:
        """Check `dt` against :meth:`max_stable_dt` for `method`.

        Parameters
        ----------
        dt : float
        method : str

        Returns
        -------
        StabilityResult
            ``number`` and ``limit`` are expressed as step sizes here.
        """
        limit = self.max_stable_dt(method)
        return StabilityResult(number=float(dt), limit=float(limit), stable=bool(dt <= limit), scheme=method, quantity="dt")

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate the semi-discrete right-hand side ``dU/dt`` at `state`."""
        return self._rhs_njit(np.asarray(state, dtype=np.float64), t, self.params)

    def _integrate(self, t_final: float, dt: Optional[float], method: str, **kwargs):
        if method == "rk4":
            if dt is None:
                raise ValueError("dt is required for method='rk4'")
            n_steps, dt = _step_count(t_final, dt)
            return rk4_integrate(self._rhs_njit, self.state0, 0.0, dt, n_steps, self.params)
        if method == "dopri5":
            dt0 = dt if dt is not None else t_final / 1000.0
            return dopri5_integrate(self._rhs_njit, self.state0, 0.0, t_final, dt0, self.params, **kwargs)
        raise ValueError(f"Unknown method '{method}' for {type(self).__name__}")

    def solve(self, t_final: float, dt: Optional[float] = None, method: str = "rk4", save_every: int = 1, **kwargs) -> PDESolution:
        """Integrate the semi-discrete system from ``t = 0`` to `t_final`.

        Parameters
        ----------
        t_final : float
        dt : float, optional
            Fixed step (required for ``"rk4"``), shrunk if needed so a whole
            number of steps lands exactly on `t_final`; initial step
            attempt for ``"dopri5"``.
        method : str
            ``"rk4"`` or ``"dopri5"`` (subclasses may add more, e.g.
            ``"leapfrog"``).
        save_every : int
            Keep every `save_every`-th time sample (the last one is always kept).
        **kwargs
            Forwarded to :func:`mathematicskit.integrators.dopri5_integrate`.

        Returns
        -------
        PDESolution
            ``extra["stability"]`` holds :meth:`stability` for the fixed
            step `dt` (omitted for ``"dopri5"``, which controls its own step).
        """
        out = self._integrate(t_final, dt, method, **kwargs)
        ts, states = out[0], out[1]
        keep = _thinned_indices(len(ts), save_every)
        solution = PDESolution(t=ts[keep], x=self.x, y=self.y, u=self._to_field(states[keep]), method=method)
        if len(out) > 2:
            solution.extra["v"] = self._to_field(out[2][keep])
        if method != "dopri5" and dt is not None:
            solution.extra["stability"] = self.stability(_step_count(t_final, dt)[1], method)
        return solution


def _step_count(t_final: float, dt: float) -> tuple[int, float]:
    """Whole number of steps reaching `t_final` exactly, and the (never larger) step that does so."""
    ratio = t_final / dt
    n_steps = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else int(np.ceil(ratio))
    n_steps = max(n_steps, 1)
    return n_steps, t_final / n_steps


def _thinned_indices(n: int, save_every: int) -> np.ndarray:
    """Indices ``0, k, 2k, ...`` of `n` samples, always including the last one."""
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    keep = np.arange(0, n, save_every)
    if keep[-1] != n - 1:
        keep = np.append(keep, n - 1)
    return keep

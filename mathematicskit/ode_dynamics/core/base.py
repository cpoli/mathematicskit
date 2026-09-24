"""Abstract base classes and result containers for mathematicskit.ode_dynamics.

:class:`FlowSystem` wraps an autonomous ODE ``dy/dt = f(y, t)`` built as
a standalone ``@njit`` callback (the first-class-function pattern
documented in physicskit's ``classical/core/base_system.py``: a
module-level factory closes over the system's numeric parameters and
returns a dispatcher, never a bound method), and integrates it via
:mod:`mathematicskit.integrators`. :class:`FixedPointResult` is the stable
result type for linear-stability analysis
(:mod:`mathematicskit.ode_dynamics.systems.stability`);
:class:`LyapunovFunctionResult` and :class:`BendixsonResult` hold the
outputs of Lyapunov's direct method and Bendixson's negative criterion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from mathematicskit.integrators import dopri5_integrate, rk4_integrate

__all__ = ["OdeTrajectory", "FlowSystem", "FixedPointResult", "LyapunovFunctionResult", "BendixsonResult"]


@dataclass
class OdeTrajectory:
    """Container for the output of a :meth:`FlowSystem.integrate` call."""

    t: np.ndarray
    """ndarray, shape (n_steps + 1,): Time samples."""

    y: np.ndarray
    """ndarray, shape (n_steps + 1, dim): State trajectory."""

    method: str = ""
    """str: Integrator used (``"rk4"`` or ``"dopri5"``)."""

    extra: dict = field(default_factory=dict)
    """dict: Free-form diagnostics slot."""


class FlowSystem(ABC):
    """Common base for an autonomous flow ``dy/dt = f(y, t)``.

    Concrete subclasses must set ``self._rhs_njit`` (an ``@njit``
    dispatcher, signature ``(state, t, params) -> dstate``, the
    :data:`mathematicskit.integrators.RHSFunc` convention) and ``self.params``
    in ``__init__``, and call ``super().__init__(state0)``.
    """

    _rhs_njit = None
    params: np.ndarray = np.empty(0)

    def __init__(self, state0):
        self.state = np.asarray(state0, dtype=np.float64)
        self.t = 0.0

    @abstractmethod
    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Plain-Python evaluation of ``dy/dt`` (thin wrapper around ``self._rhs_njit``)."""

    def reset(self, state0=None, t0: float = 0.0):
        """Reset state and clock; see e.g. physicskit's identical pattern."""
        if state0 is not None:
            self.state = np.asarray(state0, dtype=np.float64)
        self.t = t0
        return self.state

    def integrate(self, t_span, dt=None, method: str = "rk4", **kwargs) -> OdeTrajectory:
        """Integrate the flow forward in time.

        Parameters
        ----------
        t_span : tuple of float
            ``(t0, t1)``.
        dt : float, optional
            Fixed step (required for ``method="rk4"``); initial step
            attempt for ``method="dopri5"`` (defaults to ``(t1-t0)/1000``).
        method : {"rk4", "dopri5"}
        **kwargs
            Forwarded to :func:`mathematicskit.integrators.dopri5_integrate`.

        Returns
        -------
        OdeTrajectory
        """
        t0, t1 = t_span
        if method == "rk4":
            if dt is None:
                raise ValueError("dt is required for method='rk4'")
            n_steps = int(round((t1 - t0) / dt))
            ts, ys = rk4_integrate(self._rhs_njit, self.state, t0, dt, n_steps, self.params)
        elif method == "dopri5":
            dt0 = dt if dt is not None else (t1 - t0) / 1000.0
            ts, ys = dopri5_integrate(self._rhs_njit, self.state, t0, t1, dt0, self.params, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}' for FlowSystem")
        self.state = ys[-1]
        self.t = ts[-1]
        return OdeTrajectory(t=ts, y=ys, method=method)


@dataclass
class FixedPointResult:
    """Container for a 2D fixed point's linear-stability classification."""

    location: np.ndarray
    """ndarray, shape (2,): Fixed-point coordinates."""

    jacobian: np.ndarray
    """ndarray, shape (2, 2): Jacobian evaluated at the fixed point."""

    eigenvalues: np.ndarray
    """ndarray, shape (2,), complex: Eigenvalues of the Jacobian."""

    classification: str = ""
    """str: One of ``"stable node"``, ``"unstable node"``, ``"saddle"``,
    ``"stable spiral"``, ``"unstable spiral"``, ``"center"``,
    ``"degenerate node"``."""

    stable: bool = False
    """bool: Whether both eigenvalues have negative real part."""


@dataclass
class LyapunovFunctionResult:
    """Container for a quadratic Lyapunov function ``V(x) = x^T P x``."""

    P: np.ndarray
    """ndarray, shape (n, n): Symmetric solution of ``A^T P + P A = -Q``."""

    Q: np.ndarray
    """ndarray, shape (n, n): Symmetric positive-definite right-hand side used."""

    positive_definite: bool = False
    """bool: Whether ``P`` is positive definite, i.e. whether ``V`` certifies
    asymptotic stability of the origin (Lyapunov's theorem)."""


@dataclass
class BendixsonResult:
    """Container for Bendixson's negative criterion evaluated on a grid."""

    X: np.ndarray
    """ndarray, shape (n, n): Grid x-coordinates."""

    Y: np.ndarray
    """ndarray, shape (n, n): Grid y-coordinates."""

    divergence: np.ndarray
    """ndarray, shape (n, n): Divergence ``df/dx + dg/dy`` of the vector field ``(f, g)``."""

    rules_out_periodic_orbits: bool = False
    """bool: ``True`` if the divergence is strictly of one sign on the whole
    grid, so no closed orbit lies entirely inside the (simply connected)
    rectangle."""

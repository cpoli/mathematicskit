r"""The heat (diffusion) equation :math:`u_t = \alpha \nabla^2 u` in 1D and 2D.

Two routes to the same semi-discretization :math:`dU/dt = \alpha L U + b`
(:math:`L` the second-difference Laplacian, :math:`b` the boundary data):

* **Method of lines** (:meth:`HeatEquation1D.solve`,
  :meth:`HeatEquation2D.solve`): hand the ODE system to
  :mod:`mathematicskit.integrators` (``rk4``/``dopri5``). The system is
  stiff -- its eigenvalues reach :math:`-4\alpha/\Delta x^2` -- so RK4 is
  stable only for :math:`\Delta t \le 2.785\,\Delta x^2 / (4\alpha)`.
* **Theta-method** (:meth:`HeatEquation1D.solve_theta`): the one-step
  family :math:`(I - \theta\Delta t\,\alpha L)\,U^{n+1} = (I + (1-\theta)\Delta t\,\alpha L)\,U^n + \Delta t\,\alpha\,b`,
  with :math:`\theta = 0` explicit FTCS (stable iff :math:`r \le 1/2`),
  :math:`\theta = 1/2` Crank-Nicolson (Crank & Nicolson, 1947; second
  order in time, unconditionally stable), and :math:`\theta = 1` backward
  Euler. The implicit systems are factored once with
  :func:`scipy.sparse.linalg.splu` and reused every step.

:func:`heat_series_solution` is Fourier's (1807/1822) exact solution on a
finite rod with zero end temperatures, a sine series whose :math:`k`-th
mode decays like :math:`e^{-\alpha (k\pi/L)^2 t}`. See LeVeque, *Finite
Difference Methods for Ordinary and Partial Differential Equations*, SIAM
2007, Ch. 9, and Strauss, *Partial Differential Equations: An
Introduction*, 2nd ed., Ch. 4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.integrate as si
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from numba import njit

from mathematicskit.pde.core.base import MethodOfLinesPDE, PDESolution, _step_count, _thinned_indices
from mathematicskit.pde.systems.stability import RK4_REAL_AXIS_LIMIT, _theta_name, check_diffusion_stability
from mathematicskit.pde.utils.operators import laplacian_1d, laplacian_2d, uniform_grid

__all__ = ["HeatEquation1D", "HeatEquation2D", "fourier_sine_coefficients", "heat_series_solution"]

InitialCondition = Callable | np.ndarray


def _sample(u0: InitialCondition, *grids: np.ndarray) -> np.ndarray:
    """Evaluate a callable initial condition on a grid, or validate an array's shape."""
    if callable(u0):
        mesh = np.meshgrid(*grids, indexing="ij")
        return np.asarray(u0(*mesh), dtype=np.float64) * np.ones(mesh[0].shape)
    arr = np.asarray(u0, dtype=np.float64)
    shape = tuple(len(g) for g in grids)
    if arr.shape != shape:
        raise ValueError(f"initial condition has shape {arr.shape}, expected {shape}")
    return arr.copy()


@njit(cache=True)
def _heat1d_dirichlet_rhs(state, t, params):
    # params = [alpha / dx^2, u(left), u(right)]; state = interior values.
    k = params[0]
    n = state.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = state[i - 1] if i > 0 else params[1]
        right = state[i + 1] if i < n - 1 else params[2]
        out[i] = k * (left - 2.0 * state[i] + right)
    return out


@njit(cache=True)
def _heat1d_periodic_rhs(state, t, params):
    # params = [alpha / dx^2]; state = all n grid values, u_n == u_0.
    k = params[0]
    n = state.shape[0]
    out = np.empty(n)
    for i in range(n):
        left = state[i - 1] if i > 0 else state[n - 1]
        right = state[i + 1] if i < n - 1 else state[0]
        out[i] = k * (left - 2.0 * state[i] + right)
    return out


@njit(cache=True)
def _heat2d_rhs(state, t, params):
    # params = [alpha / dx^2, alpha / dy^2, nx_interior, ny_interior, boundary value];
    # state[i * ny + j] = u at interior point (i, j).
    kx, ky = params[0], params[1]
    nx, ny = int(params[2]), int(params[3])
    g = params[4]
    out = np.empty(nx * ny)
    for i in range(nx):
        for j in range(ny):
            c = state[i * ny + j]
            west = state[(i - 1) * ny + j] if i > 0 else g
            east = state[(i + 1) * ny + j] if i < nx - 1 else g
            south = state[i * ny + j - 1] if j > 0 else g
            north = state[i * ny + j + 1] if j < ny - 1 else g
            out[i * ny + j] = kx * (west - 2.0 * c + east) + ky * (south - 2.0 * c + north)
    return out


def _theta_march(L, b, u0, alpha, dt, n_steps, theta, save_every):
    """Advance ``dU/dt = alpha (L U + b)`` by the theta-method; return (step indices kept, states kept)."""
    identity = sp.identity(L.shape[0], format="csc")
    explicit = identity + (1.0 - theta) * dt * alpha * L
    lu = sla.splu((identity - theta * dt * alpha * L).tocsc()) if theta > 0.0 else None
    forcing = dt * alpha * b
    keep = _thinned_indices(n_steps + 1, save_every)
    kept = np.empty((len(keep), u0.shape[0]))
    u = u0.copy()
    slot = 0
    for step in range(n_steps + 1):
        if slot < len(keep) and keep[slot] == step:
            kept[slot] = u
            slot += 1
        if step < n_steps:
            rhs = explicit @ u + forcing
            u = lu.solve(rhs) if lu is not None else rhs
    return keep, kept


class HeatEquation1D(MethodOfLinesPDE):
    r"""The 1D heat equation :math:`u_t = \alpha u_{xx}` on :math:`[0, L]`.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
        Initial temperature, ``u0(x)`` or its grid values.
    length : float
        Rod length :math:`L`.
    n : int
        Number of grid points (both endpoints included for Dirichlet; the
        right endpoint omitted for periodic).
    alpha : float
        Diffusivity.
    bc : {"dirichlet", "periodic"}
    boundary_values : tuple of float
        Fixed end temperatures ``(u(0), u(L))`` for ``bc="dirichlet"``.

    Examples
    --------
    >>> import numpy as np
    >>> heat = HeatEquation1D(lambda x: np.sin(np.pi * x), n=41)
    >>> sol = heat.solve(0.1, dt=1e-4)
    >>> exact = np.sin(np.pi * sol.x) * np.exp(-np.pi**2 * 0.1)
    >>> bool(np.max(np.abs(sol.final - exact)) < 1e-3)
    True
    >>> cn = heat.solve_theta(0.1, dt=1e-3, theta=0.5)
    >>> bool(np.max(np.abs(cn.final - exact)) < 1e-3)
    True
    """

    def __init__(self, u0: InitialCondition, length: float = 1.0, n: int = 51, alpha: float = 1.0, bc: str = "dirichlet", boundary_values=(0.0, 0.0)):
        if bc not in ("dirichlet", "periodic"):
            raise ValueError(f"Unknown boundary condition '{bc}'")
        self.length, self.alpha, self.bc = float(length), float(alpha), bc
        self.x, self.dx = uniform_grid(0.0, self.length, n, periodic=bc == "periodic")
        self.boundary_values = (float(boundary_values[0]), float(boundary_values[1]))
        u_init = _sample(u0, self.x)
        if bc == "dirichlet":
            self.state0 = u_init[1:-1].copy()
            self.params = np.array([self.alpha / self.dx**2, *self.boundary_values])
            self._rhs_njit = _heat1d_dirichlet_rhs
        else:
            self.state0 = u_init
            self.params = np.array([self.alpha / self.dx**2])
            self._rhs_njit = _heat1d_periodic_rhs

    def _to_field(self, states: np.ndarray) -> np.ndarray:
        if self.bc == "periodic":
            return states.copy()
        left = np.full((states.shape[0], 1), self.boundary_values[0])
        right = np.full((states.shape[0], 1), self.boundary_values[1])
        return np.hstack([left, states, right])

    def max_stable_dt(self, method: str = "rk4") -> float:
        """Largest stable step: ``2.785 dx^2 / (4 alpha)`` for ``"rk4"``, ``dx^2 / (2 alpha)`` for ``"ftcs"``, ``inf`` for implicit schemes."""
        if method == "rk4":
            return RK4_REAL_AXIS_LIMIT * self.dx**2 / (4.0 * self.alpha)
        if method == "ftcs":
            return self.dx**2 / (2.0 * self.alpha)
        return np.inf

    def solve_theta(self, t_final: float, dt: float, theta: float = 0.5, save_every: int = 1) -> PDESolution:
        r"""March the theta-method: FTCS (``theta=0``), Crank-Nicolson (``0.5``), backward Euler (``1``).

        Parameters
        ----------
        t_final : float
        dt : float
        theta : float
            Implicitness in ``[0, 1]``.
        save_every : int
            Keep every `save_every`-th step (the last one is always kept).

        Returns
        -------
        PDESolution
            ``method`` is ``"ftcs"``, ``"crank_nicolson"``,
            ``"backward_euler"``, or ``"theta=<value>"``;
            ``extra["stability"]`` is
            :func:`~mathematicskit.pde.systems.stability.check_diffusion_stability`
            for this step.
        """
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        n_steps, dt = _step_count(t_final, dt)
        m = self.state0.shape[0]
        L = laplacian_1d(m, self.dx, bc=self.bc)
        b = np.zeros(m)
        if self.bc == "dirichlet":
            b[0] += self.boundary_values[0] / self.dx**2
            b[-1] += self.boundary_values[1] / self.dx**2
        keep, states = _theta_march(L, b, self.state0, self.alpha, dt, n_steps, float(theta), save_every)
        return PDESolution(
            t=keep * dt,
            x=self.x,
            u=self._to_field(states),
            method=_theta_name(theta),
            extra={"stability": check_diffusion_stability(self.alpha, dt, self.dx, theta=theta)},
        )


class HeatEquation2D(MethodOfLinesPDE):
    r"""The 2D heat equation :math:`u_t = \alpha (u_{xx} + u_{yy})` on a rectangle, fixed boundary temperature.

    Semi-discretized with the five-point Laplacian and integrated by the
    method of lines through :mod:`mathematicskit.integrators`
    (:meth:`solve`), or by the theta-method (:meth:`solve_theta`).

    Parameters
    ----------
    u0 : callable or ndarray, shape (nx, ny)
        Initial temperature ``u0(X, Y)`` (called on ``"ij"`` meshgrids) or its grid values.
    lengths : tuple of float
        Rectangle side lengths ``(Lx, Ly)``; the domain is ``[0, Lx] x [0, Ly]``.
    n : tuple of int
        Grid points ``(nx, ny)``, boundaries included.
    alpha : float
        Diffusivity.
    boundary_value : float
        Temperature held on the whole boundary.

    Examples
    --------
    >>> import numpy as np
    >>> heat = HeatEquation2D(lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y), n=(21, 21))
    >>> sol = heat.solve(0.02, dt=2e-4, save_every=10)
    >>> sol.u.shape
    (11, 21, 21)
    >>> exact = np.exp(-2 * np.pi**2 * 0.02)
    >>> bool(abs(sol.final[10, 10] - exact) < 5e-3)
    True
    """

    y: np.ndarray

    def __init__(self, u0: InitialCondition, lengths=(1.0, 1.0), n=(41, 41), alpha: float = 1.0, boundary_value: float = 0.0):
        self.alpha, self.boundary_value = float(alpha), float(boundary_value)
        self.x, self.dx = uniform_grid(0.0, float(lengths[0]), int(n[0]))
        self.y, self.dy = uniform_grid(0.0, float(lengths[1]), int(n[1]))
        self._interior_shape = (len(self.x) - 2, len(self.y) - 2)
        self.state0 = _sample(u0, self.x, self.y)[1:-1, 1:-1].ravel().copy()
        self.params = np.array([self.alpha / self.dx**2, self.alpha / self.dy**2, *self._interior_shape, self.boundary_value], dtype=np.float64)
        self._rhs_njit = _heat2d_rhs

    def _to_field(self, states: np.ndarray) -> np.ndarray:
        field = np.full((states.shape[0], len(self.x), len(self.y)), self.boundary_value)
        field[:, 1:-1, 1:-1] = states.reshape(states.shape[0], *self._interior_shape)
        return field

    def max_stable_dt(self, method: str = "rk4") -> float:
        """Largest stable step for ``"rk4"`` or ``"ftcs"``; ``inf`` for implicit schemes."""
        spectral_radius = 4.0 * self.alpha * (1.0 / self.dx**2 + 1.0 / self.dy**2)
        if method == "rk4":
            return RK4_REAL_AXIS_LIMIT / spectral_radius
        if method == "ftcs":
            return 2.0 / spectral_radius
        return np.inf

    def solve_theta(self, t_final: float, dt: float, theta: float = 0.5, save_every: int = 1) -> PDESolution:
        """March the theta-method on the five-point Laplacian; see :meth:`HeatEquation1D.solve_theta`.

        Returns
        -------
        PDESolution
            ``extra["stability"]`` is the step-size check from :meth:`stability`.
        """
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        n_steps, dt = _step_count(t_final, dt)
        nxi, nyi = self._interior_shape
        L = laplacian_2d(nxi, nyi, self.dx, self.dy)
        g = self.boundary_value
        b = np.zeros(self._interior_shape)
        b[0, :] += g / self.dx**2
        b[-1, :] += g / self.dx**2
        b[:, 0] += g / self.dy**2
        b[:, -1] += g / self.dy**2
        keep, states = _theta_march(L, b.ravel(), self.state0, self.alpha, dt, n_steps, float(theta), save_every)
        name = _theta_name(theta)
        return PDESolution(
            t=keep * dt, x=self.x, y=self.y, u=self._to_field(states), method=name, extra={"stability": self.stability(dt, "ftcs" if theta == 0.0 else name)}
        )


def fourier_sine_coefficients(f: Callable[[float], float], n_terms: int, length: float = 1.0) -> np.ndarray:
    r"""Fourier sine-series coefficients of `f` on :math:`[0, L]`.

    .. math:: b_k = \frac{2}{L} \int_0^L f(x) \sin\!\left(\frac{k\pi x}{L}\right) dx, \qquad k = 1, \dots, n_{\text{terms}}

    computed with :func:`scipy.integrate.quad`. See Fourier, *Theorie
    analytique de la chaleur* (1822), and Strauss, Ch. 5.1.

    Parameters
    ----------
    f : callable
    n_terms : int
    length : float

    Returns
    -------
    ndarray, shape (n_terms,)
        ``b[k - 1]`` is the coefficient of :math:`\sin(k\pi x/L)`.

    Examples
    --------
    >>> import numpy as np
    >>> b = fourier_sine_coefficients(lambda x: np.sin(2 * np.pi * x), 3)
    >>> np.round(b, 10).tolist()
    [0.0, 1.0, 0.0]
    """
    b = np.empty(n_terms)
    for k in range(1, n_terms + 1):
        integral, _ = si.quad(lambda s: f(s) * np.sin(k * np.pi * s / length), 0.0, length, limit=200)
        b[k - 1] = 2.0 / length * integral
    return b


def heat_series_solution(coefficients, x, t: float, length: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    r"""Fourier's series solution of the heat equation on a rod with zero end temperatures.

    .. math:: u(x, t) = \sum_{k \ge 1} b_k\, e^{-\alpha (k\pi/L)^2 t} \sin\!\left(\frac{k\pi x}{L}\right)

    Parameters
    ----------
    coefficients : array-like, shape (n_terms,)
        Sine coefficients of the initial condition, e.g. from
        :func:`fourier_sine_coefficients`.
    x : array-like
    t : float
    length, alpha : float

    Returns
    -------
    ndarray, same shape as `x`

    Examples
    --------
    >>> import numpy as np
    >>> u = heat_series_solution([1.0], np.array([0.5]), t=0.1)
    >>> bool(np.isclose(u[0], np.exp(-np.pi**2 * 0.1)))
    True
    """
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(coefficients, dtype=np.float64)
    k = np.arange(1, len(b) + 1)
    wavenumbers = k * np.pi / length
    decay = b * np.exp(-alpha * wavenumbers**2 * t)
    return np.sin(np.multiply.outer(x, wavenumbers)) @ decay

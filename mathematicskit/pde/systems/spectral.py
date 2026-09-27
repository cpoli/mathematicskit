r"""Spectral methods: Fourier (periodic) and Chebyshev (non-periodic) collocation.

Spectral methods differentiate the global interpolant through *all* grid
values instead of a local stencil, so for smooth functions the error
falls faster than any power of :math:`1/N` ("spectral accuracy"):

* **Fourier** (Orszag, 1969-1971; Kreiss & Oliger, 1972): on a periodic
  grid, :math:`\widehat{u'}_k = i k \hat u_k`. :func:`fourier_derivative`
  does this with :mod:`numpy.fft`; :func:`fourier_differentiation_matrix`
  builds the equivalent dense matrix (Trefethen 2000, Ch. 3), which is
  what lets :class:`BurgersEquation1D` run its pseudo-spectral right-hand
  side inside an ``@njit`` callback for :mod:`mathematicskit.integrators`.
* **Chebyshev** (Gottlieb & Orszag, 1977): collocation at the Chebyshev
  points :math:`x_k = -\cos(k\pi/N)` -- reusing
  :func:`mathematicskit.numerical_analysis.chebyshev_nodes` -- where
  node clustering at the ends avoids the Runge phenomenon.
  :func:`chebyshev_differentiation_matrix` is Trefethen's ``cheb.m``;
  :func:`chebyshev_poisson_1d` and :func:`chebyshev_poisson_2d` solve
  boundary-value problems with it, via :func:`mathematicskit.linalg.lu_solve_system`.

See Trefethen, *Spectral Methods in MATLAB*, SIAM 2000, Ch. 1-7, and
Boyd, *Chebyshev and Fourier Spectral Methods*, 2nd ed., 2001.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit

from mathematicskit.linalg import lu_solve_system
from mathematicskit.numerical_analysis import chebyshev_nodes
from mathematicskit.pde.core.base import EllipticSolution, MethodOfLinesPDE
from mathematicskit.pde.systems.heat import InitialCondition, _sample
from mathematicskit.pde.systems.stability import RK4_IMAG_AXIS_LIMIT, RK4_REAL_AXIS_LIMIT
from mathematicskit.pde.utils.operators import uniform_grid

__all__ = [
    "fourier_derivative",
    "fourier_differentiation_matrix",
    "chebyshev_differentiation_matrix",
    "chebyshev_poisson_1d",
    "chebyshev_poisson_2d",
    "BurgersEquation1D",
    "burgers_cole_hopf_solution",
]


def fourier_derivative(u, length: float = 2.0 * np.pi, order: int = 1) -> np.ndarray:
    r"""Spectral derivative of periodic grid values via the FFT.

    Multiplies each Fourier coefficient by :math:`(i k)^{\text{order}}`
    (:func:`numpy.fft.rfft` / :func:`~numpy.fft.irfft`). For an even
    number of points the unpaired Nyquist mode is zeroed for odd orders,
    so the derivative of real data stays real (Trefethen 2000, Ch. 3).

    Parameters
    ----------
    u : array-like, shape (n,)
        Samples at ``x_j = j * length / n``, ``j = 0, ..., n-1``.
    length : float
        Period.
    order : int
        Derivative order.

    Returns
    -------
    ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    >>> bool(np.allclose(fourier_derivative(np.sin(x)), np.cos(x)))
    True
    """
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    factor = (1j * k) ** order
    if n % 2 == 0 and order % 2 == 1:
        factor[-1] = 0.0
    return np.fft.irfft(factor * np.fft.rfft(u), n)


def fourier_differentiation_matrix(n: int, length: float = 2.0 * np.pi, order: int = 1) -> np.ndarray:
    r"""Dense Fourier spectral differentiation matrix on ``n`` periodic points (``n`` even).

    With :math:`h = 2\pi/n` and :math:`d = i - j`,

    .. math::

        D^{(1)}_{ij} = \tfrac12 (-1)^{d} \cot(d h / 2)\ (d \ne 0), \qquad
        D^{(2)}_{ij} = -\frac{(-1)^{d}}{2 \sin^2(d h/2)}\ (d \ne 0), \quad
        D^{(2)}_{ii} = -\frac{\pi^2}{3h^2} - \frac16,

    rescaled by :math:`(2\pi/\text{length})^{\text{order}}` (Trefethen
    2000, Ch. 3, eqs. (3.10)-(3.12)).

    Parameters
    ----------
    n : int
        Even number of grid points.
    length : float
        Period.
    order : {1, 2}

    Returns
    -------
    ndarray, shape (n, n)

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    >>> D2 = fourier_differentiation_matrix(16, order=2)
    >>> bool(np.allclose(D2 @ np.sin(3 * x), -9 * np.sin(3 * x)))
    True
    """
    if n % 2 or n < 2:
        raise ValueError("n must be a positive even integer")
    h = 2.0 * np.pi / n
    idx = np.arange(n)
    d = np.subtract.outer(idx, idx)
    off = d != 0
    sign = np.where(d % 2 == 0, 1.0, -1.0)
    D = np.zeros((n, n))
    if order == 1:
        D[off] = 0.5 * sign[off] / np.tan(d[off] * h / 2.0)
    elif order == 2:
        D[off] = -0.5 * sign[off] / np.sin(d[off] * h / 2.0) ** 2
        D[~off] = -(np.pi**2) / (3.0 * h**2) - 1.0 / 6.0
    else:
        raise ValueError("order must be 1 or 2")
    return D * (2.0 * np.pi / length) ** order


def chebyshev_differentiation_matrix(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Chebyshev collocation differentiation matrix on ``n`` Chebyshev points of ``[a, b]``.

    .. math:: D_{ij} = \frac{c_i}{c_j} \frac{(-1)^{i+j}}{x_i - x_j}\ (i \ne j), \qquad D_{ii} = -\sum_{j \ne i} D_{ij},

    with :math:`c_0 = c_{n-1} = 2` and :math:`c_i = 1` otherwise; the
    "negative sum trick" for the diagonal makes :math:`D` exact on
    constants and improves rounding (Trefethen 2000, Ch. 6, ``cheb.m``).
    Points come from :func:`mathematicskit.numerical_analysis.chebyshev_nodes`,
    in increasing order.

    Parameters
    ----------
    n : int
        Number of points (polynomial degree ``n - 1``).
    a, b : float

    Returns
    -------
    D : ndarray, shape (n, n)
    x : ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> D, x = chebyshev_differentiation_matrix(12)
    >>> bool(np.allclose(D @ x**5, 5 * x**4))  # exact for polynomials of degree < n
    True
    """
    x = chebyshev_nodes(n, a, b)
    c = np.ones(n)
    c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(n)
    dX = np.subtract.outer(x, x)
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D -= np.diag(D.sum(axis=1))
    return D, x


def chebyshev_poisson_1d(f: float | Callable, n: int = 32, a: float = -1.0, b: float = 1.0, ua: float = 0.0, ub: float = 0.0) -> EllipticSolution:
    r"""Solve :math:`u'' = f` on :math:`[a, b]` with :math:`u(a) = u_a`, :math:`u(b) = u_b` by Chebyshev collocation.

    Collocates :math:`D^2 u = f` at the interior Chebyshev points and
    moves the known boundary columns to the right-hand side (Trefethen
    2000, Ch. 7). For analytic `f` the error decays geometrically in `n`.

    Parameters
    ----------
    f : float or callable
    n : int
    a, b, ua, ub : float

    Returns
    -------
    EllipticSolution

    Examples
    --------
    >>> import numpy as np
    >>> sol = chebyshev_poisson_1d(lambda x: np.exp(x), n=16, ua=np.exp(-1), ub=np.exp(1))
    >>> bool(np.max(np.abs(sol.u - np.exp(sol.x))) < 1e-12)
    True
    """
    D, x = chebyshev_differentiation_matrix(n, a, b)
    D2 = D @ D
    f_int = np.asarray(f(x[1:-1]), dtype=np.float64) * np.ones(n - 2) if callable(f) else np.full(n - 2, float(f))
    A = D2[1:-1, 1:-1]
    rhs = f_int - D2[1:-1, 0] * ua - D2[1:-1, -1] * ub
    interior = lu_solve_system(A, rhs)
    u = np.concatenate([[ua], interior, [ub]])
    return EllipticSolution(x=x, u=u, method="chebyshev", residual_norm=float(np.linalg.norm(A @ interior - rhs)))


def chebyshev_poisson_2d(f: float | Callable, n: int = 24, a: float = -1.0, b: float = 1.0) -> EllipticSolution:
    r"""Solve :math:`u_{xx} + u_{yy} = f` on the square :math:`[a, b]^2` with :math:`u = 0` on the boundary.

    Tensor-product Chebyshev collocation: the interior operator is
    :math:`\tilde D^2 \otimes I + I \otimes \tilde D^2` (Trefethen 2000,
    Ch. 7, program ``p16``), solved densely with
    :func:`mathematicskit.linalg.lu_solve_system`.

    Parameters
    ----------
    f : float or callable
        Source term, ``f(X, Y)`` on ``"ij"`` meshgrids.
    n : int
        Chebyshev points per direction.
    a, b : float

    Returns
    -------
    EllipticSolution

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda X, Y: -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    >>> sol = chebyshev_poisson_2d(f, n=20)
    >>> X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    >>> bool(np.max(np.abs(sol.u - np.sin(np.pi * X) * np.sin(np.pi * Y))) < 1e-8)
    True
    """
    D, x = chebyshev_differentiation_matrix(n, a, b)
    D2 = (D @ D)[1:-1, 1:-1]
    identity = np.eye(n - 2)
    A = np.kron(D2, identity) + np.kron(identity, D2)
    X, Y = np.meshgrid(x[1:-1], x[1:-1], indexing="ij")
    rhs = (np.asarray(f(X, Y), dtype=np.float64) * np.ones(X.shape) if callable(f) else np.full(X.shape, float(f))).ravel()
    interior = lu_solve_system(A, rhs)
    u = np.zeros((n, n))
    u[1:-1, 1:-1] = interior.reshape(n - 2, n - 2)
    return EllipticSolution(x=x, y=x.copy(), u=u, method="chebyshev", residual_norm=float(np.linalg.norm(A @ interior - rhs)))


@njit(cache=True)
def _burgers_spectral_rhs(state, t, params):
    # params = [nu, D1.ravel(), D2.ravel()]; pseudo-spectral: products in physical space.
    n = state.shape[0]
    nu = params[0]
    D1 = params[1 : 1 + n * n].reshape((n, n))
    D2 = params[1 + n * n :].reshape((n, n))
    return -state * (D1 @ state) + nu * (D2 @ state)


class BurgersEquation1D(MethodOfLinesPDE):
    r"""Viscous Burgers' equation :math:`u_t + u\,u_x = \nu\,u_{xx}`, periodic, Fourier pseudo-spectral.

    Derivatives come from :func:`fourier_differentiation_matrix`; the
    nonlinear product :math:`u\,u_x` is formed pointwise in physical space
    (the pseudo-spectral, or collocation, approach of Orszag 1969-1971 and
    Kreiss & Oliger 1972), without dealiasing. The resulting ODE system is
    integrated by :mod:`mathematicskit.integrators` through an ``@njit``
    right-hand side that receives the flattened matrices in ``params``.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
    length : float
        Period.
    n : int
        Even number of grid points.
    nu : float
        Viscosity.

    Examples
    --------
    >>> import numpy as np
    >>> burgers = BurgersEquation1D(lambda x: np.sin(x), n=32, nu=0.1)
    >>> sol = burgers.solve(1.0, dt=1e-3, save_every=100)
    >>> bool(abs(sol.final.mean() - sol.u[0].mean()) < 1e-10)  # the mean is conserved
    True
    """

    def __init__(self, u0: InitialCondition, length: float = 2.0 * np.pi, n: int = 64, nu: float = 0.05):
        self.length, self.nu = float(length), float(nu)
        self.x, self.dx = uniform_grid(0.0, self.length, n, periodic=True)
        self.state0 = _sample(u0, self.x)
        D1 = fourier_differentiation_matrix(n, self.length, order=1)
        D2 = fourier_differentiation_matrix(n, self.length, order=2)
        self.params = np.concatenate([[self.nu], D1.ravel(), D2.ravel()])
        self._rhs_njit = _burgers_spectral_rhs

    def _to_field(self, states: np.ndarray) -> np.ndarray:
        return states.copy()

    def max_stable_dt(self, method: str = "rk4") -> float:
        r"""Linearized RK4 estimate: the smaller of the diffusive and advective limits.

        Diffusion contributes eigenvalues down to :math:`-\nu k_{\max}^2`
        and advection (frozen at :math:`\max|u_0|`) up to
        :math:`\pm i \max|u_0| k_{\max}`, with :math:`k_{\max} = \pi n / L`.
        """
        if method != "rk4":
            return np.inf
        k_max = np.pi * len(self.x) / self.length
        u_max = float(np.max(np.abs(self.state0)))
        diffusive = RK4_REAL_AXIS_LIMIT / (self.nu * k_max**2) if self.nu > 0 else np.inf
        advective = RK4_IMAG_AXIS_LIMIT / (u_max * k_max) if u_max > 0 else np.inf
        return float(min(diffusive, advective))


def burgers_cole_hopf_solution(u0, t: float, nu: float, length: float = 2.0 * np.pi, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    r"""Exact periodic solution of viscous Burgers' equation by the Hopf-Cole transformation.

    Hopf (1950) and Cole (1951) showed that :math:`u = -2\nu\,\varphi_x/\varphi`
    turns :math:`u_t + u u_x = \nu u_{xx}` into the heat equation
    :math:`\varphi_t = \nu \varphi_{xx}`, with
    :math:`\varphi(x, 0) = \exp\!\big(-\tfrac{1}{2\nu}\int_0^x u_0\big)`.
    Here the antiderivative, the heat evolution (each Fourier mode times
    :math:`e^{-\nu k^2 t}`), and :math:`\varphi_x` are all computed
    spectrally with :mod:`numpy.fft`, so the result is exact up to the
    resolution of the ``n``-point grid. :math:`\varphi` spans roughly
    :math:`e^{\pm \max|\int u_0| / (2\nu)}`, so very small `nu` exhausts
    double precision.

    Parameters
    ----------
    u0 : callable or ndarray, shape (n,)
        Periodic initial data with zero mean (so that :math:`\varphi` is periodic).
    t, nu : float
    length : float
        Period.
    n : int
        Even number of grid points.

    Returns
    -------
    x : ndarray, shape (n,)
    u : ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> x, u = burgers_cole_hopf_solution(np.sin, t=1.0, nu=0.1, n=128)
    >>> sol = BurgersEquation1D(np.sin, n=128, nu=0.1).solve(1.0, dt=1e-3)
    >>> bool(np.max(np.abs(sol.final - u)) < 1e-10)
    True
    """
    x, _ = uniform_grid(0.0, length, n, periodic=True)
    u_init = _sample(u0, x)
    if abs(u_init.mean()) > 1e-10 * max(1.0, float(np.max(np.abs(u_init)))):
        raise ValueError("u0 must have zero mean for a periodic Hopf-Cole solution")
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=length / n)
    u_hat = np.fft.rfft(u_init)
    U_hat = np.zeros_like(u_hat)
    U_hat[1:] = u_hat[1:] / (1j * k[1:])
    U = np.fft.irfft(U_hat, n)
    exponent = -(U - U.min()) / (2.0 * nu)
    phi_hat = np.fft.rfft(np.exp(exponent)) * np.exp(-nu * k**2 * t)
    phi = np.fft.irfft(phi_hat, n)
    dphi_hat = 1j * k * phi_hat
    if n % 2 == 0:
        dphi_hat[-1] = 0.0
    return x, -2.0 * nu * np.fft.irfft(dphi_hat, n) / phi

r"""CFL conditions, diffusion-number limits, and von Neumann stability analysis.

Courant, Friedrichs & Lewy (1928) showed that an explicit scheme for a
hyperbolic equation can only converge if its numerical domain of
dependence contains the true one, i.e. the Courant number
:math:`\nu = |c|\,\Delta t/\Delta x` must not exceed a scheme-dependent
limit (1 for upwind, Lax-Friedrichs, Lax-Wendroff and leapfrog-for-waves).
For diffusion the analogous dimensionless group is the diffusion number
:math:`r = \alpha\,\Delta t/\Delta x^2`, and explicit FTCS needs
:math:`r \le 1/2` in 1D.

Von Neumann's method (developed at Los Alamos in the 1940s; first
published by Crank & Nicolson, 1947, and Charney, Fjortoft & von Neumann,
1950) substitutes a single Fourier mode :math:`u_j^n = G^n e^{i j \xi}`
(:math:`\xi = k\,\Delta x`) into a linear constant-coefficient scheme; the
scheme is stable iff the amplification factor satisfies
:math:`|G(\xi)| \le 1` for every :math:`\xi \in [0, \pi]`. See LeVeque,
*Finite Difference Methods for Ordinary and Partial Differential
Equations*, SIAM 2007, Ch. 9.6 and 10.5-10.7, and Strikwerda, *Finite
Difference Schemes and Partial Differential Equations*, 2nd ed., Ch. 2.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.pde.core.base import StabilityResult

__all__ = [
    "RK4_REAL_AXIS_LIMIT",
    "RK4_IMAG_AXIS_LIMIT",
    "courant_number",
    "diffusion_number",
    "check_cfl",
    "check_diffusion_stability",
    "amplification_factor",
    "max_amplification",
    "AMPLIFICATION_SCHEMES",
]

#: Where classical RK4's stability region crosses the negative real axis:
#: the real root of :math:`z^3 + 4z^2 + 12z + 24 = 0`, from
#: :math:`R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 = 1` (about -2.7853).
RK4_REAL_AXIS_LIMIT = float(-np.real(next(r for r in np.roots([1.0, 4.0, 12.0, 24.0]) if abs(r.imag) < 1e-12)))

#: Where classical RK4's stability region crosses the imaginary axis,
#: :math:`2\sqrt{2}` (the positive root of :math:`|R(iy)| = 1`).
RK4_IMAG_AXIS_LIMIT = float(2.0 * np.sqrt(2.0))

# Courant-number limits for the advection schemes, and diffusion-number
# limits for the heat-equation schemes (1D).
_COURANT_LIMITS = {"upwind": 1.0, "godunov": 1.0, "lax_friedrichs": 1.0, "lax_wendroff": 1.0, "leapfrog": 1.0, "ftcs": 0.0}


def courant_number(c: float, dt: float, dx: float) -> float:
    r"""Courant number :math:`\nu = |c|\,\Delta t / \Delta x`.

    Parameters
    ----------
    c : float
        Wave / advection speed.
    dt, dx : float
        Time step and grid spacing.

    Returns
    -------
    float

    Examples
    --------
    >>> courant_number(2.0, 0.01, 0.04)
    0.5
    """
    return float(abs(c) * dt / dx)


def diffusion_number(alpha: float, dt: float, dx: float) -> float:
    r"""Diffusion number :math:`r = \alpha\,\Delta t / \Delta x^2`.

    Parameters
    ----------
    alpha : float
        Diffusivity.
    dt, dx : float
        Time step and grid spacing.

    Returns
    -------
    float

    Examples
    --------
    >>> round(diffusion_number(1.0, 0.001, 0.1), 12)
    0.1
    """
    return float(alpha * dt / dx**2)


def check_cfl(c: float, dt: float, dx: float, scheme: str = "upwind") -> StabilityResult:
    r"""Check the Courant-Friedrichs-Lewy condition for an advection/wave scheme.

    The limits (LeVeque 2007, Ch. 10) are :math:`\nu \le 1` for
    ``"upwind"``, ``"godunov"``, ``"lax_friedrichs"``, ``"lax_wendroff"`` and the
    ``"leapfrog"`` (velocity-Verlet) wave scheme; forward-time
    centered-space ``"ftcs"`` advection is unstable for every
    :math:`\nu > 0`.

    Parameters
    ----------
    c : float
    dt, dx : float
    scheme : {"upwind", "godunov", "lax_friedrichs", "lax_wendroff", "leapfrog", "ftcs"}

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> check_cfl(1.0, 0.009, 0.01).stable
    True
    >>> check_cfl(1.0, 0.011, 0.01).stable
    False
    """
    if scheme not in _COURANT_LIMITS:
        raise ValueError(f"Unknown scheme '{scheme}'")
    nu = courant_number(c, dt, dx)
    limit = _COURANT_LIMITS[scheme]
    return StabilityResult(number=nu, limit=limit, stable=bool(nu <= limit + 1e-12 and limit > 0.0), scheme=scheme, quantity="courant")


def check_diffusion_stability(alpha: float, dt: float, dx: float, theta: float = 0.0, ndim: int = 1) -> StabilityResult:
    r"""Check the diffusion-number limit of the theta-method for the heat equation.

    The theta-method (:math:`\theta = 0` explicit FTCS,
    :math:`\tfrac12` Crank-Nicolson, 1 backward Euler) on the standard
    second-difference Laplacian in `ndim` dimensions (equal spacing) is
    stable iff

    .. math:: r \le \frac{1}{2\,\text{ndim}\,(1 - 2\theta)} \quad (\theta < \tfrac12),

    and unconditionally stable for :math:`\theta \ge \tfrac12`
    (Strikwerda, Ch. 6.3; LeVeque, Ch. 9.6).

    Parameters
    ----------
    alpha : float
    dt, dx : float
    theta : float
        Implicitness, in ``[0, 1]``.
    ndim : int
        Number of space dimensions.

    Returns
    -------
    StabilityResult

    Examples
    --------
    >>> check_diffusion_stability(1.0, 0.004, 0.1).stable  # r = 0.4
    True
    >>> check_diffusion_stability(1.0, 0.006, 0.1).stable  # r = 0.6
    False
    >>> check_diffusion_stability(1.0, 1.0, 0.1, theta=0.5).limit
    inf
    """
    r = diffusion_number(alpha, dt, dx)
    limit = np.inf if theta >= 0.5 else 1.0 / (2.0 * ndim * (1.0 - 2.0 * theta))
    return StabilityResult(number=r, limit=float(limit), stable=bool(r <= limit + 1e-12), scheme=_theta_name(theta), quantity="diffusion")


def _theta_name(theta: float) -> str:
    return {0.0: "ftcs", 0.5: "crank_nicolson", 1.0: "backward_euler"}.get(float(theta), f"theta={theta:g}")


def _g_ftcs_heat(r, xi):
    return 1.0 - 4.0 * r * np.sin(xi / 2.0) ** 2 + 0j


def _g_backward_euler_heat(r, xi):
    return 1.0 / (1.0 + 4.0 * r * np.sin(xi / 2.0) ** 2) + 0j


def _g_crank_nicolson_heat(r, xi):
    s = 2.0 * r * np.sin(xi / 2.0) ** 2
    return (1.0 - s) / (1.0 + s) + 0j


def _g_upwind(nu, xi):
    return 1.0 - nu * (1.0 - np.exp(-1j * xi))


def _g_lax_friedrichs(nu, xi):
    return np.cos(xi) - 1j * nu * np.sin(xi)


def _g_lax_wendroff(nu, xi):
    return 1.0 - 1j * nu * np.sin(xi) - nu**2 * (1.0 - np.cos(xi))


def _g_ftcs_advection(nu, xi):
    return 1.0 - 1j * nu * np.sin(xi)


#: Schemes understood by :func:`amplification_factor`, each mapped to the
#: dimensionless number its amplification factor depends on.
AMPLIFICATION_SCHEMES = {
    "ftcs_heat": ("diffusion", _g_ftcs_heat),
    "backward_euler_heat": ("diffusion", _g_backward_euler_heat),
    "crank_nicolson_heat": ("diffusion", _g_crank_nicolson_heat),
    "upwind": ("courant", _g_upwind),
    "lax_friedrichs": ("courant", _g_lax_friedrichs),
    "lax_wendroff": ("courant", _g_lax_wendroff),
    "ftcs_advection": ("courant", _g_ftcs_advection),
}


def amplification_factor(scheme: str, number: float, xi) -> np.ndarray:
    r"""Von Neumann amplification factor :math:`G(\xi)` of a linear scheme.

    With :math:`r` the diffusion number and :math:`\nu` the Courant number
    (advection speed :math:`c > 0`), and :math:`s = \sin^2(\xi/2)`:

    ===========================  ===============================================
    ``"ftcs_heat"``              :math:`1 - 4 r s`
    ``"backward_euler_heat"``    :math:`1 / (1 + 4 r s)`
    ``"crank_nicolson_heat"``    :math:`(1 - 2 r s) / (1 + 2 r s)`
    ``"upwind"``                 :math:`1 - \nu (1 - e^{-i\xi})`
    ``"lax_friedrichs"``         :math:`\cos\xi - i \nu \sin\xi`
    ``"lax_wendroff"``           :math:`1 - i \nu \sin\xi - \nu^2 (1 - \cos\xi)`
    ``"ftcs_advection"``         :math:`1 - i \nu \sin\xi`
    ===========================  ===============================================

    See Strikwerda, Ch. 2.2, and LeVeque, Ch. 9.6 and 10.5.

    Parameters
    ----------
    scheme : str
        A key of :data:`AMPLIFICATION_SCHEMES`.
    number : float
        The scheme's diffusion number :math:`r` or Courant number :math:`\nu`.
    xi : float or array-like
        Phase angle(s) :math:`\xi = k\,\Delta x`, usually in ``[0, pi]``.

    Returns
    -------
    ndarray of complex

    Examples
    --------
    >>> import numpy as np
    >>> float(abs(amplification_factor("upwind", 1.0, np.pi)))
    1.0
    >>> float(abs(amplification_factor("ftcs_heat", 0.5, np.pi)))
    1.0
    """
    if scheme not in AMPLIFICATION_SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'; expected one of {sorted(AMPLIFICATION_SCHEMES)}")
    return np.asarray(AMPLIFICATION_SCHEMES[scheme][1](float(number), np.asarray(xi, dtype=np.float64)))


def max_amplification(scheme: str, number: float, n_xi: int = 721) -> float:
    r"""Largest :math:`|G(\xi)|` over :math:`\xi \in [0, \pi]`; the scheme is stable iff this is :math:`\le 1`.

    Parameters
    ----------
    scheme : str
    number : float
    n_xi : int
        Number of sampled phase angles.

    Returns
    -------
    float

    Examples
    --------
    >>> max_amplification("lax_wendroff", 0.8) <= 1.0
    True
    >>> max_amplification("ftcs_advection", 0.1) > 1.0
    True
    """
    xi = np.linspace(0.0, np.pi, n_xi)
    return float(np.max(np.abs(amplification_factor(scheme, number, xi))))

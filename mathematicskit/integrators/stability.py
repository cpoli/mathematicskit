r"""Linear stability regions of the integrators in :mod:`mathematicskit.integrators`.

Applied to Dahlquist's test equation :math:`y' = \lambda y`, every method
here reduces to a recurrence in :math:`z = \lambda h`:

* A one-step method multiplies :math:`y` by its *stability function*
  :math:`R(z)` each step: :math:`1 + z` for explicit Euler,
  :math:`1/(1 - z)` for backward Euler, and the degree-4 Taylor
  polynomial of :math:`e^z` for RK4.
* A :math:`k`-step Adams-Bashforth method gives a linear recurrence whose
  characteristic polynomial is :math:`\rho(\zeta) - z\,\sigma(\zeta)`, with
  :math:`\rho(\zeta) = \zeta^k - \zeta^{k-1}` and
  :math:`\sigma(\zeta) = \sum_j \beta_j \zeta^{k-1-j}`.

The method is *absolutely stable* at :math:`z` when :math:`|R(z)| \le 1`,
or, for a multistep method, when every root :math:`\zeta` of the
characteristic polynomial satisfies :math:`|\zeta| \le 1`. The set of
such :math:`z` is the stability region; a decaying mode is computed
without blowing up only when :math:`\lambda h` lies inside it. The
region's boundary for a multistep method is traced by the *boundary
locus* :math:`z(\theta) = \rho(e^{i\theta})/\sigma(e^{i\theta})`. See
Hairer & Wanner, *Solving ODEs II*, 2nd ed., sec. IV.2 and V.1, and
LeVeque, *Finite Difference Methods for ODEs and PDEs* (SIAM, 2007),
Ch. 7.
"""

from __future__ import annotations

import numpy as np

from mathematicskit.integrators.multistep import ADAMS_BASHFORTH_COEFFICIENTS

__all__ = ["STABILITY_METHODS", "stability_function", "is_absolutely_stable", "adams_bashforth_boundary_locus"]

_ONE_STEP = {
    "euler": lambda z: 1.0 + z,
    "implicit_euler": lambda z: 1.0 / (1.0 - z),
    "rk4": lambda z: 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0,
}
_MULTISTEP = {f"adams_bashforth{k}": k for k in (2, 3, 4)}

#: Method names accepted by :func:`is_absolutely_stable`.
STABILITY_METHODS = tuple(_ONE_STEP) + tuple(_MULTISTEP)

_ROOT_TOL = 1e-9
"""Slack on :math:`|\\zeta| \\le 1`, so points on the boundary itself
(where a root lies exactly on the unit circle) are not lost to rounding."""


def stability_function(method: str, z) -> np.ndarray:
    r"""Stability function :math:`R(z)` of a one-step method.

    Parameters
    ----------
    method : {"euler", "implicit_euler", "rk4"}
    z : complex or array-like of complex
        :math:`\lambda h` values.

    Returns
    -------
    ndarray of complex
        :math:`R(z)`, the factor one step multiplies :math:`y` by on
        :math:`y' = \lambda y`.

    Examples
    --------
    >>> import numpy as np
    >>> complex(stability_function("euler", -0.5))
    (0.5+0j)
    >>> # RK4 reproduces e^z to fourth order.
    >>> bool(abs(stability_function("rk4", -0.1) - np.exp(-0.1)) < 1e-7)
    True
    """
    if method not in _ONE_STEP:
        raise ValueError(f"unknown one-step method {method!r}; expected one of {sorted(_ONE_STEP)}")
    with np.errstate(divide="ignore", invalid="ignore"):  # backward Euler's pole at z = 1
        return np.asarray(_ONE_STEP[method](np.asarray(z, dtype=complex)))


def _adams_bashforth_root_moduli(order: int, z: np.ndarray) -> np.ndarray:
    """Largest root modulus of rho(zeta) - z sigma(zeta), batched over z."""
    beta = ADAMS_BASHFORTH_COEFFICIENTS[order - 1, :order]
    flat = z.ravel()
    # Monic characteristic polynomial zeta^k + c_1 zeta^(k-1) + ... + c_k,
    # with c_1 = -1 - z beta_0 and c_{j+1} = -z beta_j; its roots are the
    # eigenvalues of the companion matrix, computed for every z at once.
    coeffs = -flat[:, None] * beta[None, :]
    coeffs[:, 0] -= 1.0
    companion = np.zeros((flat.size, order, order), dtype=complex)
    companion[:, 0, :] = -coeffs
    idx = np.arange(order - 1)
    companion[:, idx + 1, idx] = 1.0
    return np.abs(np.linalg.eigvals(companion)).max(axis=1).reshape(z.shape)


def is_absolutely_stable(method: str, z) -> np.ndarray:
    r"""Whether ``method`` is absolutely stable at each :math:`z = \lambda h`.

    Parameters
    ----------
    method : str
        One of :data:`STABILITY_METHODS`: ``"euler"``, ``"implicit_euler"``,
        ``"rk4"``, or ``"adams_bashforth2"`` to ``"adams_bashforth4"``.
    z : complex or array-like of complex

    Returns
    -------
    ndarray of bool, same shape as `z`
        True inside the stability region (boundary included).

    Examples
    --------
    >>> is_absolutely_stable("euler", [-1.0, -2.5]).tolist()
    [True, False]
    >>> # AB2's real stability interval is [-1, 0], half of Euler's.
    >>> is_absolutely_stable("adams_bashforth2", [-0.9, -1.1]).tolist()
    [True, False]
    >>> bool(is_absolutely_stable("implicit_euler", -1e6))
    True
    """
    z = np.asarray(z, dtype=complex)
    if method in _ONE_STEP:
        return np.abs(stability_function(method, z)) <= 1.0 + _ROOT_TOL
    if method in _MULTISTEP:
        return _adams_bashforth_root_moduli(_MULTISTEP[method], z) <= 1.0 + _ROOT_TOL
    raise ValueError(f"unknown method {method!r}; expected one of {list(STABILITY_METHODS)}")


def adams_bashforth_boundary_locus(order: int, n_points: int = 721) -> np.ndarray:
    r"""Boundary locus :math:`z(\theta) = \rho(e^{i\theta})/\sigma(e^{i\theta})` of Adams-Bashforth.

    Every point where the characteristic polynomial has a root on the
    unit circle lies on this curve, so the stability region's boundary
    is part of it (for orders 1 and 2 it is all of it).

    Parameters
    ----------
    order : int
        1 to 4 (order 1 is explicit Euler).
    n_points : int
        Number of :math:`\theta` samples on :math:`[0, 2\pi]`.

    Returns
    -------
    ndarray of complex, shape (n_points,)

    Examples
    --------
    >>> import numpy as np
    >>> locus = adams_bashforth_boundary_locus(1, n_points=5)
    >>> np.allclose(locus, np.exp(1j * np.linspace(0, 2 * np.pi, 5)) - 1.0)
    True
    >>> # AB2's locus crosses the negative real axis at z = -1 (theta = pi).
    >>> round(float(adams_bashforth_boundary_locus(2, n_points=3)[1].real), 12)
    -1.0
    """
    if order not in (1, 2, 3, 4):
        raise ValueError("order must be 1, 2, 3 or 4")
    beta = ADAMS_BASHFORTH_COEFFICIENTS[order - 1, :order]
    zeta = np.exp(1j * np.linspace(0.0, 2.0 * np.pi, n_points))
    rho = zeta**order - zeta ** (order - 1)
    sigma = sum(beta[j] * zeta ** (order - 1 - j) for j in range(order))
    return rho / sigma

r"""The Kuramoto model of coupled phase oscillators and its synchronization
transition.

See Kuramoto (1975) and Strogatz, "From Kuramoto to Crawford: exploring
the onset of synchronization in populations of coupled oscillators,"
Physica D 143 (2000), 1-20.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["KuramotoModel", "kuramoto_order_parameter", "kuramoto_lorentzian_order_parameter"]


@lru_cache(maxsize=64)
def _make_kuramoto_rhs(K: float):
    # params holds the natural frequencies omega_i, so one compiled
    # callback serves every population of the same coupling strength.
    @njit(cache=True)
    def rhs(state, t, params):
        n = state.shape[0]
        c = 0.0
        s = 0.0
        for j in range(n):
            c += np.cos(state[j])
            s += np.sin(state[j])
        c /= n
        s /= n
        out = np.empty(n)
        for i in range(n):
            # (K/N) sum_j sin(theta_j - theta_i) = K (s cos(theta_i) - c sin(theta_i))
            out[i] = params[i] + K * (s * np.cos(state[i]) - c * np.sin(state[i]))
        return out

    return rhs


class KuramotoModel(FlowSystem):
    r"""Kuramoto model of :math:`N` all-to-all coupled phase oscillators.

    .. math:: \dot\theta_i = \omega_i + \frac{K}{N}\sum_{j=1}^N \sin(\theta_j - \theta_i)
              = \omega_i + K r \sin(\psi - \theta_i),

    where :math:`r e^{i\psi} = N^{-1}\sum_j e^{i\theta_j}` is the complex
    order parameter (:func:`kuramoto_order_parameter`). The mean-field
    form makes each step :math:`O(N)` rather than :math:`O(N^2)`.

    Parameters
    ----------
    theta0 : array-like, shape (N,)
        Initial phases.
    omegas : array-like, shape (N,)
        Natural frequencies (passed to the integrator as ``params``).
    K : float
        Coupling strength.

    Examples
    --------
    >>> import numpy as np
    >>> system = KuramotoModel([0.0, 1.0, 2.0], omegas=[0.0, 0.0, 0.0], K=1.0)
    >>> result = system.integrate((0.0, 50.0), dt=1e-2, method="rk4")
    >>> round(float(kuramoto_order_parameter(result.y[-1])[0]), 6)  # identical oscillators lock
    1.0
    """

    def __init__(self, theta0, omegas, K: float = 1.0):
        self.K = float(K)
        omegas = np.asarray(omegas, dtype=np.float64)
        theta0 = np.asarray(theta0, dtype=np.float64)
        if omegas.shape != theta0.shape:
            raise ValueError("theta0 and omegas must have the same shape")
        self._rhs_njit = _make_kuramoto_rhs(self.K)
        self.params = omegas
        super().__init__(theta0)

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        z = np.mean(np.exp(1j * state))
        return self.params + self.K * np.abs(z) * np.sin(np.angle(z) - state)


def kuramoto_order_parameter(theta):
    r"""Kuramoto order parameter :math:`r e^{i\psi} = N^{-1}\sum_j e^{i\theta_j}`.

    :math:`r = 0` for phases spread uniformly around the circle and
    :math:`r = 1` for perfect phase locking.

    Parameters
    ----------
    theta : array-like, shape (..., N)
        Phases; leading axes (e.g. time) are preserved.

    Returns
    -------
    r, psi : ndarray, shape (...)
        Coherence and mean phase.

    Examples
    --------
    >>> import numpy as np
    >>> r, psi = kuramoto_order_parameter(np.zeros(5))
    >>> float(r)
    1.0
    >>> r, _ = kuramoto_order_parameter(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    >>> bool(r < 1e-12)
    True
    """
    z = np.mean(np.exp(1j * np.asarray(theta, dtype=np.float64)), axis=-1)
    return np.abs(z), np.angle(z)


def kuramoto_lorentzian_order_parameter(K, gamma: float):
    r"""Steady-state coherence for Lorentzian natural frequencies (:math:`N \to \infty`).

    For :math:`g(\omega) = \gamma / [\pi(\gamma^2 + \omega^2)]`,
    Kuramoto's self-consistency equation can be solved exactly:
    incoherence (:math:`r = 0`) is the only state below the critical
    coupling :math:`K_c = 2/(\pi g(0)) = 2\gamma`, and above it

    .. math:: r = \sqrt{1 - K_c / K}.

    Parameters
    ----------
    K : float or array-like
        Coupling strength.
    gamma : float
        Half-width of the Lorentzian frequency distribution.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> kuramoto_lorentzian_order_parameter(4.0, gamma=1.0)  # K_c = 2
    0.7071067811865476
    >>> kuramoto_lorentzian_order_parameter(1.0, gamma=1.0)
    0.0
    """
    K = np.asarray(K, dtype=np.float64)
    kc = 2.0 * gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(K > kc, np.sqrt(np.maximum(1.0 - kc / K, 0.0)), 0.0)
    return float(r) if r.ndim == 0 else r

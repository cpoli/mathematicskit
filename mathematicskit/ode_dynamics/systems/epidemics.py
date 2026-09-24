r"""The Kermack-McKendrick SIR epidemic model and its closed-form
final-size and peak-prevalence relations.

See Kermack & McKendrick (1927) and Hethcote, "The Mathematics of
Infectious Diseases," SIAM Review 42(4) (2000), 599-653, Sec. 2.
Populations are expressed as fractions of a fixed total (``S + I + R =
1``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numba import njit
from scipy.optimize import brentq

from mathematicskit.ode_dynamics.core.base import FlowSystem

__all__ = ["SIRModel", "sir_final_size", "sir_peak_infected"]


@lru_cache(maxsize=64)
def _make_sir_rhs(beta: float, gamma: float):
    @njit(cache=True)
    def rhs(state, t, params):
        s, i = state[0], state[1]
        out = np.empty(3)
        infection = beta * s * i
        recovery = gamma * i
        out[0] = -infection
        out[1] = infection - recovery
        out[2] = recovery
        return out

    return rhs


class SIRModel(FlowSystem):
    r"""Kermack-McKendrick SIR model (population fractions).

    :math:`\dot S = -\beta SI`, :math:`\dot I = \beta SI - \gamma I`,
    :math:`\dot R = \gamma I`. The basic reproduction number
    :math:`R_0 = \beta/\gamma` sets the epidemic threshold: infections
    initially grow only if :math:`R_0 S_0 > 1`.

    Parameters
    ----------
    state0 : array-like, shape (3,)
        Initial fractions ``(S0, I0, R0)``.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.

    Examples
    --------
    >>> system = SIRModel([0.99, 0.01, 0.0], beta=0.3, gamma=0.1)
    >>> result = system.integrate((0.0, 100.0), dt=0.1, method="rk4")
    >>> round(float(result.y[-1].sum()), 12)  # S + I + R is conserved
    1.0
    """

    def __init__(self, state0, beta: float = 0.3, gamma: float = 0.1):
        self.beta, self.gamma = float(beta), float(gamma)
        self._rhs_njit = _make_sir_rhs(self.beta, self.gamma)
        self.params = np.empty(0)
        super().__init__(state0)

    @property
    def r0(self) -> float:
        """Basic reproduction number ``beta / gamma``."""
        return self.beta / self.gamma

    def rhs(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        s, i = state[0], state[1]
        return np.array([-self.beta * s * i, self.beta * s * i - self.gamma * i, self.gamma * i])


def sir_final_size(r0: float, s0: float = 1.0, i0: float = 0.0) -> float:
    r"""Fraction of the population still susceptible after the epidemic.

    Dividing :math:`\dot S` by :math:`\dot R` and integrating gives
    :math:`S = S_0 e^{-R_0 (R - R_{\text{init}})}`. Since :math:`I \to 0`,
    the limit :math:`S_\infty` solves the Kermack-McKendrick final-size
    equation (with :math:`R_{\text{init}} = 1 - S_0 - I_0`)

    .. math:: \ln\frac{S_0}{S_\infty} = R_0\,(S_0 + I_0 - S_\infty),

    solved here with :func:`scipy.optimize.brentq` on :math:`(0, S_0)`.

    Parameters
    ----------
    r0 : float
        Basic reproduction number ``beta / gamma``.
    s0, i0 : float
        Initial susceptible and infected fractions.

    Returns
    -------
    float
        :math:`S_\infty`.

    Examples
    --------
    >>> s_inf = sir_final_size(2.0)  # a fully susceptible population
    >>> round(s_inf, 4)
    0.2032
    >>> round(1 - s_inf, 4)  # fraction ever infected
    0.7968
    """
    if r0 <= 0 or s0 <= 0:
        raise ValueError("r0 and s0 must be positive")

    def g(s):
        return np.log(s0 / s) - r0 * (s0 + i0 - s)

    if i0 <= 0.0:
        if r0 * s0 <= 1.0:
            return float(s0)  # sub-threshold with no seed: S_inf = S_0 is the only root
        # S = S_0 is a trivial root; bracket the epidemic root just below it.
        upper = s0 * (1.0 - 1e-9)
    else:
        upper = s0
    return float(brentq(g, 1e-300, upper, xtol=1e-15))


def sir_peak_infected(r0: float, s0: float, i0: float) -> float:
    r"""Peak infected fraction of the SIR model.

    :math:`I + S - \ln(S)/R_0` is conserved, and :math:`I` peaks when
    :math:`S = 1/R_0`, giving

    .. math:: I_{\max} = I_0 + S_0 - \frac{1}{R_0}\left(1 + \ln(R_0 S_0)\right)

    when :math:`R_0 S_0 > 1` (otherwise :math:`I` only decreases and
    :math:`I_{\max} = I_0`).

    Parameters
    ----------
    r0 : float
    s0, i0 : float
        Initial susceptible and infected fractions.

    Returns
    -------
    float

    Examples
    --------
    >>> round(sir_peak_infected(3.0, 0.99, 0.01), 4)
    0.3038
    """
    if r0 * s0 <= 1.0:
        return float(i0)
    return float(i0 + s0 - (1.0 + np.log(r0 * s0)) / r0)

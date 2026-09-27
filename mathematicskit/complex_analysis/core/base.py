r"""Result containers for mathematicskit.complex_analysis.

:class:`Contour` is the shared description of a piecewise-smooth path
:math:`\gamma : [t_0, t_1] \to \mathbb{C}` consumed by every contour
integral, winding number, residue, and argument-principle routine in
``systems/``. The remaining dataclasses bundle the related arrays a
single computation produces, so visualizers get a stable interface
instead of bare tuples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Contour", "CauchyRiemannResult", "ResidueTheoremResult", "MappedGrid", "DomainColoringResult", "LaurentSeriesResult", "MobiusClassification"]


@dataclass
class Contour:
    r"""A piecewise-smooth path :math:`\gamma(t)` with derivative :math:`\gamma'(t)`.

    ``gamma`` and ``dgamma`` must accept scalar or array ``t``.
    ``breakpoints`` are the sorted parameter values where
    :math:`\gamma'` may jump (a polygon's corners); the path runs from
    ``breakpoints[0]`` to ``breakpoints[-1]`` and is smooth on each
    piece in between.
    """

    gamma: Callable
    dgamma: Callable
    breakpoints: np.ndarray

    def points(self, n: int = 400) -> np.ndarray:
        """``n`` points :math:`\\gamma(t)` at evenly spaced parameter values, endpoints included."""
        return np.asarray(self.gamma(np.linspace(self.breakpoints[0], self.breakpoints[-1], n)), dtype=complex)


@dataclass
class CauchyRiemannResult:
    r"""Partial derivatives of :math:`u = \operatorname{Re} f` and :math:`v = \operatorname{Im} f` at one point."""

    u_x: float
    u_y: float
    v_x: float
    v_y: float

    @property
    def residual(self) -> float:
        r"""float: :math:`\max(|u_x - v_y|, |u_y + v_x|)`, zero exactly when the Cauchy-Riemann equations hold."""
        return max(abs(self.u_x - self.v_y), abs(self.u_y + self.v_x))


@dataclass
class ResidueTheoremResult:
    r"""Both sides of the residue theorem :math:`\oint_\gamma f = 2\pi i \sum_k n(\gamma, z_k)\operatorname{Res}(f, z_k)`."""

    integral: complex
    """complex: The contour integral, computed directly."""

    poles: np.ndarray
    residues: np.ndarray
    """ndarray: :math:`\\operatorname{Res}(f, z_k)` for each pole, computed on a small circle."""

    winding_numbers: np.ndarray
    """ndarray of int: :math:`n(\\gamma, z_k)`, the number of times the contour winds around each pole."""

    predicted: complex
    """complex: :math:`2\\pi i \\sum_k n(\\gamma, z_k)\\operatorname{Res}(f, z_k)`."""


@dataclass
class MappedGrid:
    """Horizontal and vertical grid lines in the z-plane and their images under a map ``f``.

    Each array has one row per grid line.
    """

    horizontal: np.ndarray
    vertical: np.ndarray
    horizontal_image: np.ndarray
    vertical_image: np.ndarray


@dataclass
class DomainColoringResult:
    r"""A sampled complex function :math:`w = f(z)` and its domain-coloring image."""

    z: np.ndarray
    w: np.ndarray
    rgb: np.ndarray
    """ndarray: ``z.shape + (3,)`` RGB values in :math:`[0, 1]`; hue encodes :math:`\\arg w`, brightness :math:`\\log_2|w|`."""


@dataclass
class LaurentSeriesResult:
    r"""Laurent coefficients :math:`a_k` of :math:`f(z) = \sum_k a_k (z - z_0)^k` on the circle :math:`|z - z_0| = r`."""

    center: complex
    radius: float
    orders: np.ndarray
    """ndarray of int: The powers :math:`k`, from ``-n_max`` to ``n_max``."""

    coefficients: np.ndarray

    def coefficient(self, k: int) -> complex:
        """:math:`a_k` for one power ``k``."""
        return complex(self.coefficients[int(k) - int(self.orders[0])])

    def __call__(self, z):
        """Evaluate the truncated series at ``z``."""
        z = np.asarray(z, dtype=complex)[..., np.newaxis]
        return np.sum(self.coefficients * (z - self.center) ** self.orders, axis=-1)


@dataclass
class MobiusClassification:
    r"""The conjugacy type and fixed points of a Möbius transformation."""

    kind: str
    """str: ``"identity"``, ``"elliptic"``, ``"parabolic"``, ``"hyperbolic"``, or ``"loxodromic"``."""

    trace_squared: complex
    """complex: :math:`(a + d)^2/(ad - bc)`, invariant under conjugation and rescaling."""

    fixed_points: np.ndarray
    """ndarray of complex: Solutions of :math:`T(z) = z`; ``inf`` stands for the point at infinity."""

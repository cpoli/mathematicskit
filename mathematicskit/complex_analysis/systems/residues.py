r"""Residues, the residue theorem, and the argument principle.

Residues are computed as :math:`\operatorname{Res}(f, z_0) =
\frac{1}{2\pi i}\oint_{|z - z_0| = r} f(z)\,dz` on a small circle with
the periodic trapezoidal rule, which picks out the Laurent coefficient
:math:`a_{-1}` up to aliasing terms :math:`a_{-1 \pm N} r^{\pm N}` and
so converges geometrically in the number of points :math:`N`
(L. N. Trefethen and J. A. C. Weideman, "The exponentially convergent
trapezoidal rule," *SIAM Review* 56 (2014), 385-458, Sec. 16). Adaptive
quadrature, by contrast, loses accuracy to cancellation when the
integrand is large on a small circle around a high-order pole. See A.-L. Cauchy, "Sur un nouveau genre de calcul analogue au calcul
infinitésimal," *Exercices de mathématiques* 1 (1826), 11-24; Ahlfors,
*Complex Analysis*, Ch. 4, Sec. 5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from mathematicskit.complex_analysis.core.base import Contour, ResidueTheoremResult
from mathematicskit.complex_analysis.systems.contours import contour_integral, winding_number

__all__ = ["residue", "residue_theorem", "argument_principle", "rouche_condition"]


def residue(f: Callable, z0: complex, radius: float = 1e-2, n_points: int = 256) -> complex:
    r"""The residue of ``f`` at an isolated singularity ``z0``.

    Parameters
    ----------
    f : callable
        Vectorized ``f(z) -> complex``.
    z0 : complex
    radius : float
        Radius of the integration circle; must be smaller than the
        distance from ``z0`` to any other singularity.
    n_points : int
        Trapezoidal-rule nodes; must exceed the order of the pole.

    Returns
    -------
    complex

    Examples
    --------
    >>> r = residue(lambda z: np.exp(z) / z**3, 0.0)  # coefficient of 1/z is 1/2!
    >>> bool(np.isclose(r, 0.5))
    True
    """
    # (1/2 pi i) oint f dz with dz = i r e^{it} dt is the mean of f(z) r e^{it} over the circle.
    step = radius * np.exp(2j * np.pi * np.arange(n_points) / n_points)
    return complex(np.mean(f(z0 + step) * step))


def residue_theorem(f: Callable, contour: Contour, poles: Sequence[complex], residue_radius: float | None = None) -> ResidueTheoremResult:
    r"""Compare :math:`\oint_\gamma f` with :math:`2\pi i \sum_k n(\gamma, z_k)\operatorname{Res}(f, z_k)`.

    Parameters
    ----------
    f : callable
        Vectorized and meromorphic, with singularities only at ``poles``.
    contour : Contour
        Closed, not passing through any pole.
    poles : sequence of complex
        Every singularity of ``f`` (those outside the contour contribute
        winding number zero).
    residue_radius : float, optional
        Radius used by :func:`residue`; defaults to a quarter of the
        smallest distance between poles, capped at 0.1.

    Returns
    -------
    ResidueTheoremResult

    Examples
    --------
    >>> from mathematicskit.complex_analysis.systems.contours import circle_contour
    >>> f = lambda z: 1 / ((z - 0.5) * (z - 3))
    >>> result = residue_theorem(f, circle_contour(), [0.5, 3])
    >>> result.winding_numbers.tolist()
    [1, 0]
    >>> bool(np.isclose(result.integral, result.predicted))
    True
    """
    z = np.asarray(poles, dtype=complex)
    if residue_radius is None:
        gaps = np.abs(z[:, None] - z[None, :])[~np.eye(z.size, dtype=bool)]
        residue_radius = min(0.1, 0.25 * gaps.min()) if gaps.size else 0.1
    residues = np.array([residue(f, p, residue_radius) for p in z])
    windings = np.array([winding_number(contour, p) for p in z], dtype=int)
    return ResidueTheoremResult(
        integral=contour_integral(f, contour),
        poles=z,
        residues=residues,
        winding_numbers=windings,
        predicted=complex(2j * np.pi * np.sum(windings * residues)),
    )


def argument_principle(f: Callable, contour: Contour, fprime: Callable | None = None, n_points: int = 4000) -> int:
    r"""Zeros minus poles of ``f`` inside ``contour``, :math:`N - P = \frac{1}{2\pi i}\oint_\gamma \frac{f'(z)}{f(z)}\,dz`.

    With ``fprime`` the logarithmic derivative is integrated directly.
    Without it, :math:`N - P` is the winding number of the image curve
    :math:`f(\gamma)` about 0, read off from the unwrapped phase
    (:func:`numpy.unwrap`) of :math:`f` sampled along the contour.
    Both count with multiplicity. See Cauchy (1831, 1855), discussed in
    F. Smithies, *Cauchy and the Creation of Complex Function Theory*
    (Cambridge University Press, 1997), Ch. 6.

    Parameters
    ----------
    f : callable
        Meromorphic, with no zeros or poles on the contour.
    contour : Contour
        Positively oriented and simple.
    fprime : callable, optional
        :math:`f'`.
    n_points : int
        Samples used for the phase-unwrapping method; must resolve every
        turn of :math:`f(\gamma)` about 0.

    Returns
    -------
    int

    Examples
    --------
    >>> from mathematicskit.complex_analysis.systems.contours import circle_contour
    >>> p = lambda z: z**3 - 0.25 * z  # zeros at 0 and +-0.5
    >>> argument_principle(p, circle_contour()), argument_principle(p, circle_contour(), fprime=lambda z: 3 * z**2 - 0.25)
    (3, 3)
    """
    if fprime is not None:
        value = contour_integral(lambda z: fprime(z) / f(z), contour) / (2j * np.pi)
        return int(round(value.real))
    w = f(contour.points(n_points))
    return int(round((np.unwrap(np.angle(w))[-1] - np.angle(w[0])) / (2.0 * np.pi)))


def rouche_condition(f: Callable, g: Callable, contour: Contour, n_points: int = 4000) -> bool:
    r"""Whether :math:`|f(z) - g(z)| < |g(z)|` at every sampled point of ``contour``.

    By Rouché's theorem, when this holds, :math:`f` and :math:`g` have the
    same number of zeros inside the contour (with multiplicity), so a
    hard-to-count ``f`` can be compared with a simple dominant term
    ``g``. The check is on ``n_points`` samples, not a proof. See E.
    Rouché, "Mémoire sur la série de Lagrange," *Journal de l'École
    Polytechnique* 22 (1862), 193-224; Ahlfors, *Complex Analysis*,
    Ch. 4, Sec. 5.2.

    Parameters
    ----------
    f, g : callable
        Vectorized and holomorphic inside and on the contour.
    contour : Contour
    n_points : int

    Returns
    -------
    bool

    Examples
    --------
    >>> from mathematicskit.complex_analysis.systems.contours import circle_contour
    >>> f = lambda z: z**5 + 3 * z**2 + 1
    >>> rouche_condition(f, lambda z: 3 * z**2, circle_contour()), rouche_condition(f, lambda z: z**5, circle_contour(0, 2))
    (True, True)
    """
    z = contour.points(n_points)
    return bool(np.all(np.abs(f(z) - g(z)) < np.abs(g(z))))

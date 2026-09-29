"""mathematicskit.complex_analysis: functions of a complex variable.

The Cauchy-Riemann equations and complex derivatives (central
differences from :mod:`mathematicskit.calculus`); contours, contour
integrals, winding numbers, and Cauchy's integral formula (via
:func:`scipy.integrate.quad` with ``complex_func=True``); Laurent series
via :func:`numpy.fft.fft`; residues, the residue theorem, the argument
principle, and Rouché's theorem; conformal maps (Möbius transformations
and their classification, the Joukowski map) and their action on coordinate
grids; and domain coloring (phase portraits) of complex functions.
"""

from mathematicskit import __version__
from mathematicskit.complex_analysis.core.base import (
    CauchyRiemannResult,
    Contour,
    DomainColoringResult,
    LaurentSeriesResult,
    MappedGrid,
    MobiusClassification,
    ResidueTheoremResult,
)
from mathematicskit.complex_analysis.systems.conformal_maps import classify_mobius, joukowski_map, map_grid, mobius_transform
from mathematicskit.complex_analysis.systems.contours import cauchy_integral_formula, circle_contour, contour_integral, polygon_contour, winding_number
from mathematicskit.complex_analysis.systems.domain_coloring import domain_coloring
from mathematicskit.complex_analysis.systems.holomorphic import cauchy_riemann, complex_derivative
from mathematicskit.complex_analysis.systems.laurent import laurent_coefficients
from mathematicskit.complex_analysis.systems.residues import argument_principle, residue, residue_theorem, rouche_condition
from mathematicskit.complex_analysis.utils.grids import complex_grid

__all__ = [
    "__version__",
    "Contour",
    "CauchyRiemannResult",
    "ResidueTheoremResult",
    "MappedGrid",
    "DomainColoringResult",
    "LaurentSeriesResult",
    "MobiusClassification",
    "complex_derivative",
    "cauchy_riemann",
    "circle_contour",
    "polygon_contour",
    "contour_integral",
    "winding_number",
    "cauchy_integral_formula",
    "residue",
    "residue_theorem",
    "argument_principle",
    "rouche_condition",
    "laurent_coefficients",
    "mobius_transform",
    "classify_mobius",
    "joukowski_map",
    "map_grid",
    "domain_coloring",
    "complex_grid",
]

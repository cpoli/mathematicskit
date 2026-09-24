"""mathematicskit.abstract_algebra: group, ring, and field theory made computational.

Every algorithm here is hand-rolled from its textbook definition --
finite group/ring/field theory has no ``numpy``/``scipy`` equivalent.
Cyclic, permutation, dihedral, and quaternion group implementations
with Cayley table generation; group-property checks (order, identity,
inverses, abelian, cyclicity); subgroup and coset enumeration for small
groups; structure theory (normal subgroups and quotients, derived
series and solvability, Sylow subgroups, composition series); group
actions and Burnside orbit counting; homomorphisms and their kernels
and images; Reed-Solomon codes over ``GF(p)``; finite
field arithmetic (``GF(p)`` and ``GF(p^n)`` via irreducible polynomials
over ``GF(p)``); and polynomial ring arithmetic (addition,
multiplication, division with remainder, gcd) over :math:`\\mathbb{Z}`,
:math:`\\mathbb{Q}`, and finite fields.
"""

from mathematicskit.abstract_algebra.core.base import (
    CompositionSeriesResult,
    FiniteGroup,
    GroupPropertiesResult,
    HomomorphismResult,
    Polynomial,
    SylowResult,
)
from mathematicskit.abstract_algebra.systems.actions import count_orbits, orbits
from mathematicskit.abstract_algebra.systems.codes import rs_decode_erasures, rs_encode
from mathematicskit.abstract_algebra.systems.finite_fields import GF, find_irreducible_polynomial, is_irreducible
from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, DihedralGroup, PermutationGroup, QuaternionGroup, group_properties
from mathematicskit.abstract_algebra.systems.homomorphisms import analyze_homomorphism, homomorphism_image, homomorphism_kernel, is_homomorphism
from mathematicskit.abstract_algebra.systems.polynomial_ring import poly_add, poly_divmod, poly_gcd, poly_mul, poly_sub
from mathematicskit.abstract_algebra.systems.structure import (
    QuotientGroup,
    Subgroup,
    commutator_subgroup,
    composition_series,
    derived_series,
    elements_of_order,
    generated_subgroup,
    is_normal_subgroup,
    is_solvable,
    quotient_group,
    sylow_subgroups,
)
from mathematicskit.abstract_algebra.systems.subgroups import all_subgroups, cyclic_subgroup, left_cosets
from mathematicskit.abstract_algebra.utils.checks import is_cyclic

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FiniteGroup",
    "GroupPropertiesResult",
    "Polynomial",
    "CyclicGroup",
    "PermutationGroup",
    "group_properties",
    "is_cyclic",
    "cyclic_subgroup",
    "all_subgroups",
    "left_cosets",
    "GF",
    "is_irreducible",
    "find_irreducible_polynomial",
    "poly_add",
    "poly_sub",
    "poly_mul",
    "poly_divmod",
    "poly_gcd",
    "SylowResult",
    "CompositionSeriesResult",
    "HomomorphismResult",
    "DihedralGroup",
    "QuaternionGroup",
    "Subgroup",
    "generated_subgroup",
    "elements_of_order",
    "is_normal_subgroup",
    "QuotientGroup",
    "quotient_group",
    "commutator_subgroup",
    "derived_series",
    "is_solvable",
    "sylow_subgroups",
    "composition_series",
    "orbits",
    "count_orbits",
    "is_homomorphism",
    "homomorphism_kernel",
    "homomorphism_image",
    "analyze_homomorphism",
    "rs_encode",
    "rs_decode_erasures",
]

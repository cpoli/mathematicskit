"""mathematicskit.abstract_algebra: group, ring, and field theory made computational.

Every algorithm here is hand-rolled from its textbook definition --
finite group/ring/field theory has no ``numpy``/``scipy`` equivalent.
Cyclic and permutation group implementations with Cayley table
generation; group-property checks (order, identity, inverses, abelian,
cyclicity); subgroup and coset enumeration for small groups; finite
field arithmetic (``GF(p)`` and ``GF(p^n)`` via irreducible polynomials
over ``GF(p)``); and polynomial ring arithmetic (addition,
multiplication, division with remainder, gcd) over :math:`\\mathbb{Z}`,
:math:`\\mathbb{Q}`, and finite fields.
"""

from mathematicskit.abstract_algebra.core.base import FiniteGroup, GroupPropertiesResult, Polynomial
from mathematicskit.abstract_algebra.systems.finite_fields import GF, find_irreducible_polynomial, is_irreducible
from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, PermutationGroup, group_properties
from mathematicskit.abstract_algebra.systems.polynomial_ring import poly_add, poly_divmod, poly_gcd, poly_mul, poly_sub
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
]

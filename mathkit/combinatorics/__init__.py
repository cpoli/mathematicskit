"""mathkit.combinatorics: combinatorial mathematics.

Permutation and combination counting via ``scipy.special.perm``/``comb``
(sequence *generation*, which scipy doesn't provide, is mathkit's own
thin wrapper around ``itertools``); binomial and multinomial
coefficients via ``scipy.special.comb``, with a hand-rolled Pascal's-
triangle build kept only as a pedagogical illustration of the
recurrence; integer partitions (the partition function via Euler's
pentagonal-number recurrence, and full enumeration) and Young/Ferrers
diagrams; the inclusion-exclusion principle and the classic derangement-
counting application; and Stirling numbers (first and second kind),
Catalan numbers, and Bell numbers with their combinatorial
interpretations -- all hand-rolled, with no scipy/numpy equivalent.
"""

from mathkit.combinatorics.core.base import YoungDiagram
from mathkit.combinatorics.systems.counting import combinations_count, generate_combinations, generate_permutations, multinomial_coefficient, permutations_count
from mathkit.combinatorics.systems.inclusion_exclusion import derangement_count, union_size_inclusion_exclusion
from mathkit.combinatorics.systems.partitions import integer_partitions, partition_function
from mathkit.combinatorics.systems.pascals_triangle import pascals_triangle
from mathkit.combinatorics.systems.special_numbers import catalan_number, stirling_first_kind, stirling_second_kind
from mathkit.combinatorics.utils.bell_number import bell_number

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "YoungDiagram",
    "permutations_count",
    "combinations_count",
    "multinomial_coefficient",
    "generate_permutations",
    "generate_combinations",
    "pascals_triangle",
    "partition_function",
    "integer_partitions",
    "union_size_inclusion_exclusion",
    "derangement_count",
    "stirling_first_kind",
    "stirling_second_kind",
    "catalan_number",
    "bell_number",
]

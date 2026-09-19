"""Concrete combinatorial algorithms."""

from mathematicskit.combinatorics.systems.counting import (
    combinations_count,
    generate_combinations,
    generate_permutations,
    multinomial_coefficient,
    permutations_count,
)
from mathematicskit.combinatorics.systems.inclusion_exclusion import derangement_count, union_size_inclusion_exclusion
from mathematicskit.combinatorics.systems.partitions import integer_partitions, partition_function
from mathematicskit.combinatorics.systems.pascals_triangle import pascals_triangle
from mathematicskit.combinatorics.systems.special_numbers import catalan_number, stirling_first_kind, stirling_second_kind

__all__ = [
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
]

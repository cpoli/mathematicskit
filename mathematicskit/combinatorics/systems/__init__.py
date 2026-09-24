"""Concrete combinatorial algorithms."""

from mathematicskit.combinatorics.systems.counting import (
    combinations_count,
    generate_combinations,
    generate_permutations,
    multinomial_coefficient,
    permutations_count,
)
from mathematicskit.combinatorics.systems.designs import are_orthogonal, cyclic_latin_square, is_latin_square, orthogonal_latin_square_pair
from mathematicskit.combinatorics.systems.extremal import (
    count_triangle_free_colorings,
    has_monochromatic_triangle,
    longest_decreasing_subsequence,
    longest_increasing_subsequence,
)
from mathematicskit.combinatorics.systems.inclusion_exclusion import derangement_count, union_size_inclusion_exclusion
from mathematicskit.combinatorics.systems.matching import hall_condition, maximum_matching
from mathematicskit.combinatorics.systems.necklaces import count_bracelets, count_necklaces
from mathematicskit.combinatorics.systems.partitions import integer_partitions, partition_function
from mathematicskit.combinatorics.systems.pascals_triangle import pascals_triangle
from mathematicskit.combinatorics.systems.sequences import bernoulli_numbers, domino_tilings, fibonacci, gray_code, sum_of_powers
from mathematicskit.combinatorics.systems.special_numbers import catalan_number, stirling_first_kind, stirling_second_kind
from mathematicskit.combinatorics.systems.trees import count_labeled_trees, prufer_decode, prufer_encode

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
    "are_orthogonal",
    "cyclic_latin_square",
    "is_latin_square",
    "orthogonal_latin_square_pair",
    "count_triangle_free_colorings",
    "has_monochromatic_triangle",
    "longest_decreasing_subsequence",
    "longest_increasing_subsequence",
    "hall_condition",
    "maximum_matching",
    "count_bracelets",
    "count_necklaces",
    "bernoulli_numbers",
    "domino_tilings",
    "fibonacci",
    "gray_code",
    "sum_of_powers",
    "count_labeled_trees",
    "prufer_decode",
    "prufer_encode",
]

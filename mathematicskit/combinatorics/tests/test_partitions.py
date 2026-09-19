"""Tests for integer partitions against closed-form/known results."""

import pytest

from mathematicskit.combinatorics.core.base import YoungDiagram
from mathematicskit.combinatorics.systems.partitions import integer_partitions, partition_function


@pytest.mark.parametrize("n,expected", [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5), (5, 7), (10, 42)])
def test_partition_function_matches_known_values(n, expected):
    assert partition_function(n) == expected


def test_partition_function_matches_enumeration_count():
    for n in range(0, 15):
        assert partition_function(n) == len(integer_partitions(n))


def test_integer_partitions_of_four():
    assert integer_partitions(4) == [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]


def test_every_partition_sums_to_n():
    for partition in integer_partitions(8):
        assert sum(partition) == 8
        assert partition == sorted(partition, reverse=True)


def test_conjugate_of_conjugate_is_identity():
    diagram = YoungDiagram([5, 3, 3, 1])
    assert diagram.conjugate().conjugate().parts == diagram.parts


def test_conjugate_preserves_n():
    diagram = YoungDiagram([4, 2, 1])
    assert diagram.conjugate().n == diagram.n


def test_young_diagram_rejects_non_decreasing_parts():
    with pytest.raises(ValueError):
        YoungDiagram([1, 2, 3])


def test_young_diagram_rejects_nonpositive_parts():
    with pytest.raises(ValueError):
        YoungDiagram([3, 0, 1])

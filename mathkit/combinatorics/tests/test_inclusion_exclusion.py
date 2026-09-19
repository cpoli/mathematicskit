"""Tests for the inclusion-exclusion principle and derangement counting
against closed-form/known results."""

import itertools
import math

from mathkit.combinatorics.systems.inclusion_exclusion import derangement_count, union_size_inclusion_exclusion


def test_union_size_matches_python_set_union():
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    c = {4, 5, 6, 7}
    assert union_size_inclusion_exclusion([a, b, c]) == len(a | b | c)


def test_union_size_disjoint_sets_is_plain_sum():
    a, b, c = {1, 2}, {3, 4}, {5, 6}
    assert union_size_inclusion_exclusion([a, b, c]) == 6


def test_union_size_single_set():
    a = {1, 2, 3}
    assert union_size_inclusion_exclusion([a]) == 3


def test_derangement_known_values():
    assert derangement_count(0) == 1
    assert derangement_count(1) == 0
    assert derangement_count(2) == 1
    assert derangement_count(3) == 2
    assert derangement_count(4) == 9


def test_derangement_matches_brute_force():
    for n in range(6):
        brute_force = sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))
        assert derangement_count(n) == brute_force


def test_derangement_ratio_approaches_1_over_e():
    n = 15
    ratio = derangement_count(n) / math.factorial(n)
    assert abs(ratio - 1.0 / math.e) < 1e-6

"""Tests for R(3,3) = 6 and the Erdős-Szekeres theorem."""

import random
from itertools import combinations

import pytest

from mathematicskit.combinatorics.systems.extremal import (
    count_triangle_free_colorings,
    has_monochromatic_triangle,
    longest_decreasing_subsequence,
    longest_increasing_subsequence,
)


def test_ramsey_r33_is_six():
    assert count_triangle_free_colorings(5) > 0
    assert count_triangle_free_colorings(6) == 0


def test_pentagon_coloring_avoids_monochromatic_triangles():
    coloring = {(i, j): int((j - i) % 5 in (1, 4)) for i, j in combinations(range(5), 2)}
    assert not has_monochromatic_triangle(5, coloring)


def _is_strictly_increasing(xs):
    return all(a < b for a, b in zip(xs, xs[1:]))


def _brute_force_lis_length(seq):
    best = 0
    for r in range(len(seq) + 1):
        for idx in combinations(range(len(seq)), r):
            if _is_strictly_increasing([seq[i] for i in idx]):
                best = max(best, r)
    return best


@pytest.mark.parametrize("seed", range(10))
def test_lis_matches_brute_force(seed):
    rng = random.Random(seed)
    seq = [rng.randint(0, 20) for _ in range(12)]
    lis = longest_increasing_subsequence(seq)
    assert _is_strictly_increasing(lis)
    assert len(lis) == _brute_force_lis_length(seq)


@pytest.mark.parametrize("seed", range(20))
def test_erdos_szekeres_bound(seed):
    r = s = 4
    rng = random.Random(seed)
    seq = rng.sample(range(1000), (r - 1) * (s - 1) + 1)
    assert len(longest_increasing_subsequence(seq)) >= r or len(longest_decreasing_subsequence(seq)) >= s


def test_erdos_szekeres_bound_is_sharp():
    seq = [7, 8, 9, 4, 5, 6, 1, 2, 3]  # (r-1)(s-1) = 9 terms with no monotone run of length 4
    assert len(longest_increasing_subsequence(seq)) == 3
    assert len(longest_decreasing_subsequence(seq)) == 3


def test_empty_sequence():
    assert longest_increasing_subsequence([]) == []

"""Tests for bipartite matching and Hall's condition."""

import random

import pytest

from mathematicskit.combinatorics.systems.matching import hall_condition, maximum_matching


def test_matching_is_valid_and_covers_left_side():
    adjacency = {"a": [1, 2], "b": [1], "c": [2, 3]}
    matching = maximum_matching(adjacency)
    assert len(matching) == 3
    assert len(set(matching.values())) == 3
    assert all(v in adjacency[u] for u, v in matching.items())


def test_hall_violation_is_reported():
    result = hall_condition({"a": [1], "b": [1], "c": [2]})
    assert not result.satisfied
    assert result.violating_subset == frozenset({"a", "b"})
    assert len(result.matching) == 2


@pytest.mark.parametrize("seed", range(30))
def test_hall_condition_iff_perfect_matching(seed):
    rng = random.Random(seed)
    adjacency = {u: rng.sample(range(6), rng.randint(0, 3)) for u in range(5)}
    result = hall_condition(adjacency)
    assert result.satisfied == (len(result.matching) == len(adjacency))

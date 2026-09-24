"""Tests for Burnside's lemma against direct orbit enumeration and the
classical necklace counts."""

from itertools import product

import pytest

from mathematicskit.abstract_algebra.systems.actions import count_orbits, orbits
from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, DihedralGroup


def _rotate(g, word):
    return word[g:] + word[:g]


@pytest.mark.parametrize("n,expected", [(3, 4), (4, 6), (5, 8), (6, 14)])
def test_binary_necklace_counts(n, expected):
    words = ["".join(w) for w in product("01", repeat=n)]
    assert count_orbits(CyclicGroup(n), words, _rotate) == expected


def test_burnside_matches_direct_orbit_enumeration():
    words = ["".join(w) for w in product("RGB", repeat=4)]
    assert count_orbits(CyclicGroup(4), words, _rotate) == len(orbits(CyclicGroup(4), words, _rotate)) == 24


def test_bracelets_under_dihedral_group():
    n = 6
    d = DihedralGroup(n)
    words = [tuple(w) for w in product((0, 1), repeat=n)]
    # D_6 permutes bead positions: position i moves to g[i].
    act = lambda g, w: tuple(w[g.index(i)] for i in range(n))  # noqa: E731
    assert count_orbits(d, words, act) == 13


def test_orbits_partition_the_points():
    words = ["".join(w) for w in product("01", repeat=4)]
    result = orbits(CyclicGroup(4), words, _rotate)
    assert sorted(x for orbit in result for x in orbit) == sorted(words)

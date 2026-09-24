"""Tests for Pólya necklace and bracelet counts against brute force."""

from itertools import product

import pytest

from mathematicskit.combinatorics.systems.necklaces import count_bracelets, count_necklaces


def _canonical(word, reflections):
    n = len(word)
    variants = [word[i:] + word[:i] for i in range(n)]
    if reflections:
        rev = word[::-1]
        variants += [rev[i:] + rev[:i] for i in range(n)]
    return min(variants)


@pytest.mark.parametrize("n,k", [(n, k) for n in range(1, 8) for k in (2, 3)])
def test_counts_match_brute_force(n, k):
    words = list(product(range(k), repeat=n))
    assert count_necklaces(n, k) == len({_canonical(w, False) for w in words})
    assert count_bracelets(n, k) == len({_canonical(w, True) for w in words})

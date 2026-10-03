"""Tests for conjugacy classes and character tables."""

import numpy as np
import pytest

from mathematicskit.abstract_algebra.systems.characters import character_table, conjugacy_classes
from mathematicskit.abstract_algebra.systems.groups import CyclicGroup, DihedralGroup, PermutationGroup, QuaternionGroup

GROUPS = [
    CyclicGroup(6),
    PermutationGroup(3),
    DihedralGroup(4),
    QuaternionGroup(),
    DihedralGroup(5),
    PermutationGroup(4),
    PermutationGroup(5, generators=[(1, 2, 0, 3, 4), (0, 1, 3, 4, 2)]),  # A_5
]


def test_classes_partition_the_group():
    for g in GROUPS:
        classes = conjugacy_classes(g)
        assert classes[0] == [g.identity()]
        assert sorted(map(str, (x for c in classes for x in c))) == sorted(map(str, g.elements))
        assert all(g.order % len(c) == 0 for c in classes)


@pytest.mark.parametrize("group", GROUPS, ids=lambda g: f"order{g.order}")
def test_orthogonality_relations(group):
    result = character_table(group)
    chi = result.table
    sizes = np.array(result.class_sizes)
    n = group.order
    assert chi.shape == (len(result.classes),) * 2
    assert np.allclose((chi * sizes) @ chi.conj().T / n, np.eye(len(sizes)))  # row orthogonality
    assert np.allclose(chi.conj().T @ chi, np.diag(n / sizes))  # column orthogonality
    assert sum(d**2 for d in result.degrees) == n
    assert all(n % d == 0 for d in result.degrees)
    assert np.allclose(chi[0], 1.0)


def test_abelian_groups_have_linear_characters():
    result = character_table(CyclicGroup(6))
    assert result.degrees == [1] * 6
    values = {complex(round(z.real, 9), round(z.imag, 9)) for z in result.table.ravel()}
    sixth_roots = {complex(round(np.cos(2 * np.pi * k / 6), 9), round(np.sin(2 * np.pi * k / 6), 9)) for k in range(6)}
    assert values == sixth_roots


def test_d4_and_q8_share_a_character_table():
    d4, q8 = character_table(DihedralGroup(4)), character_table(QuaternionGroup())
    assert d4.class_sizes == q8.class_sizes == [1, 1, 2, 2, 2]
    assert sorted(map(tuple, d4.table.real.round().astype(int).tolist())) == sorted(map(tuple, q8.table.real.round().astype(int).tolist()))


def test_s4_degrees_and_a5_golden_ratio():
    assert character_table(PermutationGroup(4)).degrees == [1, 1, 2, 3, 3]
    a5 = character_table(GROUPS[-1])
    assert a5.degrees == [1, 3, 3, 4, 5]
    phi = (1 + np.sqrt(5)) / 2
    assert np.any(np.isclose(a5.table.real, phi)) and np.any(np.isclose(a5.table.real, 1 - phi))

"""Tests for the Chinese Remainder Theorem against closed-form/known solutions."""

import pytest

from mathematicskit.number_theory.systems.crt import chinese_remainder_theorem


def test_classic_three_congruence_example():
    result = chinese_remainder_theorem([2, 3, 2], [3, 5, 7])
    assert result.residue == 23
    assert result.modulus == 105


def test_solution_satisfies_every_congruence():
    remainders = [4, 7, 9]
    moduli = [11, 13, 17]
    result = chinese_remainder_theorem(remainders, moduli)
    for r, m in zip(remainders, moduli):
        assert result.residue % m == r


def test_single_congruence():
    result = chinese_remainder_theorem([5], [12])
    assert result.residue == 5
    assert result.modulus == 12


def test_rejects_non_coprime_moduli():
    with pytest.raises(ValueError):
        chinese_remainder_theorem([1, 2], [4, 6])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        chinese_remainder_theorem([1, 2], [3])

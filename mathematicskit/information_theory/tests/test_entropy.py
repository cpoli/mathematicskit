"""Tests for mathematicskit.information_theory.systems.entropy."""

import numpy as np
import pytest

from mathematicskit.information_theory import (
    binary_entropy,
    conditional_entropy,
    entropy,
    hartley_information,
    joint_entropy,
    kl_divergence,
    mutual_information,
)


@pytest.mark.parametrize("n", [1, 2, 7, 26, 1024])
def test_entropy_of_uniform_distribution_equals_hartley_information(n):
    assert entropy(np.ones(n)) == pytest.approx(hartley_information(n))
    assert entropy(np.ones(n)) == pytest.approx(np.log2(n))


def test_hartley_information_is_additive_over_length_and_base_changes_units():
    assert hartley_information(10, length=3, base=10) == pytest.approx(3.0)
    assert hartley_information([2, 4, 8]) == pytest.approx([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        hartley_information(0)


def test_entropy_bounds_and_natural_units():
    p = np.random.default_rng(0).dirichlet(np.ones(6))
    assert 0 <= entropy(p) <= np.log2(6)
    assert entropy(p, base=np.e) == pytest.approx(entropy(p) * np.log(2))
    assert entropy([3, 1]) == pytest.approx(entropy([0.75, 0.25]))
    with pytest.raises(ValueError):
        entropy([0.5, -0.5])


def test_binary_entropy_symmetry_and_maximum():
    p = np.linspace(0, 1, 101)
    h = binary_entropy(p)
    assert h == pytest.approx(h[::-1])
    assert h.max() == pytest.approx(1.0)
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(0.3) == pytest.approx(entropy([0.3, 0.7]))
    with pytest.raises(ValueError):
        binary_entropy(1.5)


def _random_joint(seed=1, shape=(3, 4)):
    return np.random.default_rng(seed).dirichlet(np.ones(np.prod(shape))).reshape(shape)


def test_chain_rule_and_mutual_information_identities():
    joint = _random_joint()
    hx, hy = entropy(joint.sum(axis=1)), entropy(joint.sum(axis=0))
    assert joint_entropy(joint) == pytest.approx(hx + conditional_entropy(joint))
    assert joint_entropy(joint) == pytest.approx(hy + conditional_entropy(joint.T))
    info = mutual_information(joint)
    assert info == pytest.approx(mutual_information(joint.T))
    assert info == pytest.approx(hy - conditional_entropy(joint))
    assert 0 <= info <= min(hx, hy)


def test_mutual_information_is_kl_divergence_from_product_of_marginals():
    joint = _random_joint(2)
    product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    assert mutual_information(joint) == pytest.approx(kl_divergence(joint.ravel(), product.ravel()))
    assert mutual_information(product) == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_across_binary_symmetric_channel_is_its_capacity():
    p = 0.1
    joint = 0.5 * np.array([[1 - p, p], [p, 1 - p]])
    assert mutual_information(joint) == pytest.approx(1 - binary_entropy(p))


def test_kl_divergence_gibbs_inequality_asymmetry_and_infinite_case():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
        assert kl_divergence(p, q) >= 0
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert kl_divergence([0.5, 0.5], [0.9, 0.1]) != pytest.approx(kl_divergence([0.9, 0.1], [0.5, 0.5]))
    assert kl_divergence([0.5, 0.5], [1.0, 0.0]) == np.inf
    with pytest.raises(ValueError):
        kl_divergence([0.5, 0.5], [1 / 3, 1 / 3, 1 / 3])

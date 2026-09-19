"""Tests for Lyapunov exponent estimation against closed-form/known
results: the logistic map at r=4 (exactly conjugate to a tent map with
exponent ln 2), a stable fixed point's negative exponent, and a simple
linear flow's exponent equal to its own decay rate."""

import numpy as np
import pytest

from mathkit.fractals_chaos.systems.lyapunov import lyapunov_exponent_1d_map, lyapunov_exponent_flow


def test_logistic_map_r4_lyapunov_exponent_is_ln2():
    f = lambda x: 4.0 * x * (1.0 - x)
    fprime = lambda x: 4.0 - 8.0 * x
    lam = lyapunov_exponent_1d_map(f, fprime, x0=0.4, n_iterations=200000)
    assert lam == pytest.approx(np.log(2.0), abs=0.02)


def test_stable_fixed_point_has_negative_exponent():
    """The logistic map at r=2.5 has a stable fixed point x*=1-1/r=0.6,
    where |f'(x*)| < 1, so nearby orbits converge (negative exponent)."""
    f = lambda x: 2.5 * x * (1.0 - x)
    fprime = lambda x: 2.5 - 5.0 * x
    lam = lyapunov_exponent_1d_map(f, fprime, x0=0.3, n_iterations=5000)
    assert lam < 0.0


def test_linear_flow_exponent_matches_its_own_decay_rate():
    rhs = lambda x: -2.0 * x
    lam = lyapunov_exponent_flow(rhs, np.array([1.0]), n_steps=20000)
    assert lam == pytest.approx(-2.0, abs=0.1)


def test_expanding_flow_has_positive_exponent():
    """A short observation window is used deliberately: this shadow-trajectory
    estimator assumes the reference trajectory stays within a bounded region
    (as any chaotic attractor does); an unboundedly expanding system like this
    one eventually grows past the point where the fixed absolute shadow
    separation (1e-8) is resolvable in floating point relative to it."""
    rhs = lambda x: 1.5 * x
    lam = lyapunov_exponent_flow(rhs, np.array([1e-3]), dt=0.001, n_steps=2000)
    assert lam == pytest.approx(1.5, abs=0.05)

"""Tests for the SIR model against its closed-form final size and peak."""

import numpy as np
import pytest

from mathematicskit.ode_dynamics.systems.epidemics import SIRModel, sir_final_size, sir_peak_infected


@pytest.fixture(scope="module")
def epidemic():
    system = SIRModel([0.999, 0.001, 0.0], beta=0.5, gamma=0.2)
    return system, system.integrate((0.0, 600.0), dt=0.05, method="rk4")


def test_population_is_conserved(epidemic):
    _, result = epidemic
    np.testing.assert_allclose(result.y.sum(axis=1), 1.0, atol=1e-12)


def test_final_size_matches_integration(epidemic):
    system, result = epidemic
    assert result.y[-1, 0] == pytest.approx(sir_final_size(system.r0, 0.999, 0.001), abs=1e-8)


def test_peak_infected_matches_integration(epidemic):
    system, result = epidemic
    assert result.y[:, 1].max() == pytest.approx(sir_peak_infected(system.r0, 0.999, 0.001), abs=1e-5)


def test_final_size_satisfies_its_equation():
    r0 = 2.5
    s_inf = sir_final_size(r0)
    assert np.log(1.0 / s_inf) == pytest.approx(r0 * (1.0 - s_inf))


def test_below_threshold_no_epidemic():
    assert sir_final_size(0.8) == 1.0
    assert sir_peak_infected(0.8, 0.99, 0.01) == 0.01


def test_final_size_rejects_nonpositive_r0():
    with pytest.raises(ValueError):
        sir_final_size(0.0)

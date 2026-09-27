"""Tests for mathematicskit.pde.systems.wave and mathematicskit.pde.systems.advection."""

import numpy as np
import pytest

from mathematicskit.pde import AdvectionEquation1D, WaveEquation1D, dalembert_solution


def _pluck(x):
    return np.maximum(0.0, 1.0 - np.abs(x - 0.3) / 0.1)


def test_leapfrog_at_unit_courant_reproduces_dalembert_with_fixed_ends():
    string = WaveEquation1D(_pluck, n=101, c=1.0)
    sol = string.solve(1.37, dt=string.dx, method="leapfrog")
    exact = dalembert_solution(_pluck, sol.x, 1.37, length=1.0)
    assert np.max(np.abs(sol.final - exact)) < 1e-10


def test_initial_velocity_matches_dalembert_and_separation_of_variables():
    c, t = 2.0, 0.3
    string = WaveEquation1D(np.zeros(81), v0=lambda x: np.sin(np.pi * x), n=81, c=c)
    exact_sov = np.sin(np.pi * string.x) * np.sin(np.pi * c * t) / (np.pi * c)
    exact_da = dalembert_solution(lambda s: 0 * s, string.x, t, c=c, g_antiderivative=lambda s: -np.cos(np.pi * s) / np.pi, length=1.0)
    assert np.allclose(exact_da, exact_sov, atol=1e-12)
    for method in ("leapfrog", "rk4"):
        sol = string.solve(t, dt=0.4 * string.dx / c, method=method)
        assert np.max(np.abs(sol.final - exact_sov)) < 1e-3


def test_dalembert_on_whole_line_splits_bump():
    bump = lambda s: np.exp(-50 * s**2)
    x = np.array([-2.0, 2.0])
    assert np.allclose(dalembert_solution(bump, x, 1.0, c=2.0), 0.5)


def test_leapfrog_energy_is_bounded_while_rk4_dissipates():
    string = WaveEquation1D(_pluck, n=101)
    dt = 0.9 * string.dx
    leap = string.solve(20.0, dt=dt, method="leapfrog", save_every=10)
    rk = string.solve(20.0, dt=dt, method="rk4", save_every=10)
    e_leap, e_rk = string.energy(leap), string.energy(rk)
    assert np.max(np.abs(e_leap / e_leap[0] - 1.0)) < 0.05
    assert e_rk[-1] < 0.95 * e_rk[0]
    assert rk.extra["v"].shape == rk.u.shape


def test_leapfrog_blows_up_beyond_cfl_limit():
    string = WaveEquation1D(_pluck, n=51)
    assert string.max_stable_dt("leapfrog") == pytest.approx(string.dx)
    assert string.max_stable_dt("rk4") == pytest.approx(np.sqrt(2) * string.dx)
    sol = string.solve(3.0, dt=1.05 * string.dx, method="leapfrog")
    assert not sol.extra["stability"].stable
    assert np.max(np.abs(sol.final)) > 10.0


def test_wave_dopri5_and_missing_dt():
    string = WaveEquation1D(lambda x: np.sin(np.pi * x), n=41)
    sol = string.solve(0.5, method="dopri5", rtol=1e-8, atol=1e-10)
    assert np.allclose(sol.final, 0.0, atol=2e-3)  # quarter period: displacement passes through zero
    with pytest.raises(ValueError):
        string.solve(0.5, method="leapfrog")


def _smooth(x):
    return np.sin(2 * np.pi * x)


@pytest.mark.parametrize("scheme", ["upwind", "lax_friedrichs", "lax_wendroff"])
def test_stable_advection_schemes_track_exact_solution(scheme):
    adv = AdvectionEquation1D(_smooth, n=400, c=1.0)
    sol = adv.solve(1.0, dt=0.8 * adv.dx, scheme=scheme)
    assert sol.extra["stability"].stable
    assert np.max(np.abs(sol.final - adv.exact(1.0))) < 0.1


def _advection_error(scheme, n):
    adv = AdvectionEquation1D(_smooth, n=n)
    return np.max(np.abs(adv.solve(0.5, dt=0.5 * adv.dx, scheme=scheme).final - adv.exact(0.5)))


def test_upwind_is_first_order_and_lax_wendroff_second_order():
    for scheme, expected in (("upwind", 1.0), ("lax_wendroff", 2.0)):
        e1, e2 = _advection_error(scheme, 100), _advection_error(scheme, 200)
        assert np.log2(e1 / e2) == pytest.approx(expected, abs=0.15)


def test_upwind_at_unit_courant_is_exact_shift():
    adv = AdvectionEquation1D(_smooth, n=50, c=-1.0)
    sol = adv.solve(0.3, dt=adv.dx, scheme="upwind")
    assert np.allclose(sol.final, adv.exact(0.3))


@pytest.mark.parametrize("scheme, dt_factor", [("ftcs", 0.5), ("upwind", 1.2), ("lax_wendroff", 1.2)])
def test_unstable_advection_grows(scheme, dt_factor):
    adv = AdvectionEquation1D(lambda x: np.exp(-100 * (x - 0.5) ** 2), n=100)
    sol = adv.solve(2.0, dt=dt_factor * adv.dx, scheme=scheme)
    assert not sol.extra["stability"].stable
    assert np.max(np.abs(sol.final)) > 10.0


def test_advection_exact_accepts_array_initial_data_and_rejects_bad_scheme():
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    adv = AdvectionEquation1D(np.sin(2 * np.pi * x), n=64)
    assert np.allclose(adv.exact(0.25), np.sin(2 * np.pi * (x - 0.25)), atol=1e-2)
    with pytest.raises(ValueError):
        adv.solve(0.1, dt=0.01, scheme="beam_warming")
    with pytest.raises(ValueError):
        adv.step(adv.u0, 0.01, scheme="beam_warming")

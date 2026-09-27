"""Tests for finite elements, multigrid, Godunov's method, and the Hopf-Cole solution."""

import numpy as np
import pytest

from mathematicskit.pde import (
    BurgersConservationLaw1D,
    BurgersEquation1D,
    burgers_cole_hopf_solution,
    burgers_riemann_solution,
    fem_poisson_1d,
    multigrid_poisson_2d,
    solve_poisson_1d,
    solve_poisson_2d,
    total_variation,
)


def _sine_source(X, Y):
    return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)


def test_fem_is_nodally_exact_on_nonuniform_mesh_for_polynomial_load():
    nodes = np.sort(np.concatenate([[0.0, 1.0], np.random.default_rng(3).uniform(0, 1, 9)]))
    sol = fem_poisson_1d(lambda x: 12 * x**2, nodes=nodes, ua=1.0, ub=2.0)  # u = x^4 + 1
    assert np.allclose(sol.u, nodes**4 + 1, atol=1e-12)
    assert sol.method == "fem"


def test_fem_matches_finite_differences_on_uniform_mesh_with_constant_load():
    fem = fem_poisson_1d(2.0, n_elements=10)
    fd = solve_poisson_1d(2.0, n=11)
    assert np.allclose(fem.u, fd.u)


def test_fem_nodal_error_is_only_quadrature_error():
    f = lambda x: -(np.pi**2) * np.sin(np.pi * x)
    coarse = fem_poisson_1d(f, n_elements=8)
    assert np.max(np.abs(coarse.u - np.sin(np.pi * coarse.x))) < 1e-7  # far below the O(h^2) = 1e-2 interior error
    exact_load = fem_poisson_1d(f, n_elements=8, quad_points=8)
    assert np.max(np.abs(exact_load.u - np.sin(np.pi * exact_load.x))) < 1e-12


def test_fem_rejects_bad_meshes():
    with pytest.raises(ValueError):
        fem_poisson_1d(1.0, nodes=np.array([0.0, 0.5, 0.4, 1.0]))
    with pytest.raises(ValueError):
        fem_poisson_1d(1.0, nodes=np.array([0.0, 1.0]))


def test_multigrid_matches_direct_solve():
    g = lambda X, Y: X * Y
    mg = multigrid_poisson_2d(_sine_source, n=33, boundary=g, tol=1e-10)
    direct = solve_poisson_2d(_sine_source, n=(33, 33), boundary=g)
    assert mg.solver_result.converged
    assert np.allclose(mg.u, direct.u, atol=1e-9)
    assert mg.u[0, -1] == pytest.approx(0.0) and mg.u[-1, -1] == pytest.approx(1.0)


def test_multigrid_cycle_count_is_independent_of_grid_size():
    counts = [multigrid_poisson_2d(_sine_source, n=n).solver_result.iterations for n in (17, 33, 65, 129)]
    assert max(counts) - min(counts) <= 1
    history = multigrid_poisson_2d(_sine_source, n=65).solver_result.residual_history
    assert np.all(history[1:] / history[:-1] < 0.3)  # every V-cycle cuts the residual by a fixed factor


def test_multigrid_beats_jacobi_by_orders_of_magnitude():
    mg = multigrid_poisson_2d(_sine_source, n=17).solver_result.iterations
    jacobi = solve_poisson_2d(_sine_source, n=(17, 17), method="jacobi", tol=1e-8).solver_result.iterations
    assert jacobi > 20 * mg


def test_multigrid_rejects_bad_grid_and_handles_zero_problem():
    with pytest.raises(ValueError):
        multigrid_poisson_2d(0.0, n=30)
    zero = multigrid_poisson_2d(0.0, n=9)
    assert zero.solver_result.iterations == 0 and np.all(zero.u == 0.0)


def _step(x):
    return np.where(x < 0.0, 1.0, 0.0)


def test_godunov_shock_moves_at_rankine_hugoniot_speed():
    law = BurgersConservationLaw1D(_step, x_range=(-1, 1), n=400)
    sol = law.solve(1.0, dt=0.004, scheme="godunov")
    shock_position = sol.x[np.argmin(np.abs(sol.final - 0.5))]
    assert shock_position == pytest.approx(0.5, abs=2 * law.dx)
    assert sol.extra["stability"].stable


def test_godunov_rarefaction_converges_to_entropy_solution():
    errors = []
    for n in (100, 200, 400):
        law = BurgersConservationLaw1D(lambda x: np.where(x < 0.0, -0.5, 1.0), x_range=(-1, 1), n=n)
        sol = law.solve(0.5, dt=0.8 * law.dx, scheme="godunov")
        errors.append(np.mean(np.abs(sol.final - burgers_riemann_solution(-0.5, 1.0, sol.x, 0.5))))
    assert errors[2] < errors[1] < errors[0]


def test_godunov_is_tvd_and_lax_wendroff_oscillates():
    law = BurgersConservationLaw1D(_step, x_range=(-1, 1), n=200)
    godunov = law.solve(0.8, dt=0.005, scheme="godunov")
    lw = law.solve(0.8, dt=0.005, scheme="lax_wendroff")
    tv = [total_variation(u) for u in godunov.u]
    assert np.all(np.diff(tv) <= 1e-12)
    assert godunov.final.max() <= 1.0 + 1e-12 and godunov.final.min() >= -1e-12
    assert lw.final.max() > 1.1  # overshoot at the shock (Godunov's theorem)
    assert total_variation(lw.final) > total_variation(law.u0)


@pytest.mark.parametrize("scheme", ["godunov", "lax_friedrichs", "lax_wendroff"])
def test_periodic_conservation_law_conserves_mass(scheme):
    law = BurgersConservationLaw1D(lambda x: 0.5 + np.sin(2 * np.pi * x), n=100, bc="periodic")
    sol = law.solve(0.5, dt=0.5 * law.dx, scheme=scheme)
    assert np.sum(sol.final) * law.dx == pytest.approx(np.sum(law.u0) * law.dx, abs=1e-12)


def test_conservation_law_rejects_bad_inputs():
    with pytest.raises(ValueError):
        BurgersConservationLaw1D(_step, bc="reflecting")
    law = BurgersConservationLaw1D(_step)
    with pytest.raises(ValueError):
        law.solve(0.1, dt=0.001, scheme="roe")
    with pytest.raises(ValueError):
        law.step(law.u0, 0.001, scheme="roe")


def test_cole_hopf_agrees_with_pseudo_spectral_solver():
    x, u = burgers_cole_hopf_solution(lambda s: np.sin(s) + 0.5 * np.cos(2 * s), t=0.8, nu=0.15, n=128)
    sol = BurgersEquation1D(lambda s: np.sin(s) + 0.5 * np.cos(2 * s), n=128, nu=0.15).solve(0.8, dt=1e-3)
    assert np.allclose(x, sol.x)
    assert np.max(np.abs(u - sol.final)) < 1e-10


def test_cole_hopf_initial_time_returns_initial_data_and_requires_zero_mean():
    x, u = burgers_cole_hopf_solution(np.sin, t=0.0, nu=0.2, n=64)
    assert np.allclose(u, np.sin(x), atol=1e-12)
    with pytest.raises(ValueError):
        burgers_cole_hopf_solution(lambda s: 1.0 + np.sin(s), t=0.1, nu=0.1)

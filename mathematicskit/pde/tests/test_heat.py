"""Tests for mathematicskit.pde.systems.heat against Fourier's exact solutions."""

import numpy as np
import pytest

from mathematicskit.pde import HeatEquation1D, HeatEquation2D, fourier_sine_coefficients, heat_series_solution


def _mode(x):
    return np.sin(np.pi * x)


def _exact_mode(x, t, alpha=1.0):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def _max_error(sol, t, alpha=1.0):
    return np.max(np.abs(sol.final - _exact_mode(sol.x, t, alpha)))


def test_method_of_lines_rk4_matches_decaying_mode():
    heat = HeatEquation1D(_mode, n=41, alpha=0.5)
    sol = heat.solve(0.2, dt=2e-4)
    assert sol.t[-1] == pytest.approx(0.2)
    assert _max_error(sol, 0.2, alpha=0.5) < 5e-4
    assert sol.u[:, 0] == pytest.approx(0.0) and sol.u[:, -1] == pytest.approx(0.0)
    assert sol.extra["stability"].stable


def test_method_of_lines_dopri5_matches_decaying_mode():
    sol = HeatEquation1D(_mode, n=41).solve(0.1, method="dopri5", rtol=1e-8, atol=1e-10)
    assert sol.method == "dopri5"
    assert "stability" not in sol.extra
    assert _max_error(sol, 0.1) < 1e-3


def test_method_of_lines_spatial_error_is_second_order():
    errors = []
    for n in (11, 21, 41):
        heat = HeatEquation1D(_mode, n=n)
        errors.append(_max_error(heat.solve(0.05, dt=0.2 * heat.max_stable_dt("rk4")), 0.05))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8)


def test_rk4_blows_up_above_its_stability_limit():
    heat = HeatEquation1D(lambda x: np.sin(np.pi * x) + 0.01 * np.sin(20 * np.pi * x), n=41)
    dt = 1.2 * heat.max_stable_dt("rk4")
    sol = heat.solve(0.2, dt=dt)
    assert not sol.extra["stability"].stable
    assert np.max(np.abs(sol.final)) > 10.0


def test_save_every_thins_output_but_keeps_last_sample():
    sol = HeatEquation1D(_mode, n=21).solve(0.01, dt=1e-4, save_every=30)
    assert sol.t[0] == 0.0 and sol.t[-1] == pytest.approx(0.01)
    assert len(sol.t) == 5  # steps 0, 30, 60, 90, and the final 100


def test_invalid_save_every_and_method_raise():
    heat = HeatEquation1D(_mode, n=11)
    with pytest.raises(ValueError):
        heat.solve(0.01, dt=1e-4, save_every=0)
    with pytest.raises(ValueError):
        heat.solve(0.01, dt=1e-4, method="euler")
    with pytest.raises(ValueError):
        heat.solve(0.01)


def test_crank_nicolson_is_second_order_in_time_and_backward_euler_first():
    # Fine grid so time error dominates; compare against the semi-discrete exact decay.
    n, t_final = 101, 0.1
    heat = HeatEquation1D(_mode, n=n)
    dx = heat.dx
    lam = -4.0 / dx**2 * np.sin(np.pi * dx / 2) ** 2  # eigenvalue of the discrete Laplacian for this mode
    semi_exact = np.sin(np.pi * heat.x) * np.exp(lam * t_final)
    for theta, expected_order in ((0.5, 2.0), (1.0, 1.0)):
        errors = [np.max(np.abs(heat.solve_theta(t_final, dt, theta=theta).final - semi_exact)) for dt in (0.01, 0.005, 0.0025)]
        orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
        assert np.allclose(orders, expected_order, atol=0.15)


def test_theta_method_names_and_stability_flags():
    heat = HeatEquation1D(_mode, n=21)
    dt = 1e-3  # r = 0.4
    assert heat.solve_theta(0.01, dt, theta=0.0).method == "ftcs"
    assert heat.solve_theta(0.01, dt, theta=0.5).method == "crank_nicolson"
    assert heat.solve_theta(0.01, dt, theta=1.0).method == "backward_euler"
    assert heat.solve_theta(0.01, dt, theta=0.3).method == "theta=0.3"
    with pytest.raises(ValueError):
        heat.solve_theta(0.01, dt, theta=1.5)


def test_ftcs_is_stable_below_half_and_blows_up_above():
    u0 = lambda x: np.sin(np.pi * x) + 0.01 * np.sin(19 * np.pi * x)
    heat = HeatEquation1D(u0, n=21)
    stable = heat.solve_theta(0.1, dt=0.45 * heat.dx**2, theta=0.0)
    unstable = heat.solve_theta(0.1, dt=0.6 * heat.dx**2, theta=0.0)
    assert stable.extra["stability"].stable and not unstable.extra["stability"].stable
    assert np.max(np.abs(stable.final)) < 1.0
    assert np.max(np.abs(unstable.final)) > 1e3


def test_crank_nicolson_stays_bounded_at_large_step():
    heat = HeatEquation1D(_mode, n=41)
    sol = heat.solve_theta(0.5, dt=0.05, theta=0.5)  # r = 80
    assert sol.extra["stability"].stable
    assert np.max(np.abs(sol.final)) <= 1.0


def test_dirichlet_boundary_values_relax_to_linear_profile():
    heat = HeatEquation1D(np.zeros(21), n=21, boundary_values=(1.0, 3.0))
    sol = heat.solve_theta(5.0, dt=0.05, theta=1.0)
    assert np.allclose(sol.final, 1.0 + 2.0 * sol.x, atol=1e-6)
    assert np.allclose(heat.solve(2.0, dt=0.2 * heat.max_stable_dt()).final, 1.0 + 2.0 * sol.x, atol=1e-4)


def test_periodic_heat_conserves_mean_and_decays_modes():
    heat = HeatEquation1D(lambda x: 1.0 + np.cos(2 * np.pi * x), n=32, bc="periodic")
    sol = heat.solve(0.05, dt=1e-4)
    assert sol.final.mean() == pytest.approx(1.0)
    assert np.allclose(sol.final, 1.0 + np.exp(-4 * np.pi**2 * 0.05) * np.cos(2 * np.pi * sol.x), atol=1e-3)
    cn = heat.solve_theta(0.05, dt=1e-3)
    assert np.allclose(cn.final, sol.final, atol=1e-3)


def test_heat_1d_rejects_bad_inputs():
    with pytest.raises(ValueError):
        HeatEquation1D(_mode, bc="neumann")
    with pytest.raises(ValueError):
        HeatEquation1D(np.zeros(5), n=10)


def test_rhs_matches_discrete_laplacian():
    heat = HeatEquation1D(_mode, n=11, alpha=2.0)
    u = heat.state0
    expected = 2.0 * (np.concatenate([[0.0], u[:-1]]) - 2 * u + np.concatenate([u[1:], [0.0]])) / heat.dx**2
    assert np.allclose(heat.rhs(u), expected)


def test_fourier_sine_coefficients_of_parabola():
    b = fourier_sine_coefficients(lambda x: x * (1 - x), 6)
    k = np.arange(1, 7)
    expected = np.where(k % 2 == 1, 8.0 / (k * np.pi) ** 3, 0.0)
    assert np.allclose(b, expected, atol=1e-10)


def test_heat_series_matches_finite_difference_solution():
    f = lambda x: x * (1 - x)
    b = fourier_sine_coefficients(f, 25)
    heat = HeatEquation1D(f, n=81)
    sol = heat.solve_theta(0.05, dt=5e-4, theta=0.5)
    assert np.allclose(sol.final, heat_series_solution(b, sol.x, 0.05), atol=1e-4)


def test_heat_2d_method_of_lines_matches_product_mode():
    u0 = lambda X, Y: np.sin(np.pi * X) * np.sin(2 * np.pi * Y / 2.0)
    heat = HeatEquation2D(u0, lengths=(1.0, 2.0), n=(21, 41), alpha=0.5)
    sol = heat.solve(0.05, dt=0.5 * heat.max_stable_dt(), save_every=50)
    X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    exact = u0(X, Y) * np.exp(-0.5 * 2 * np.pi**2 * 0.05)
    assert sol.u.shape[1:] == (21, 41)
    assert sol.y is not None and sol.y[-1] == pytest.approx(2.0)
    assert np.max(np.abs(sol.final - exact)) < 2e-3


def test_heat_2d_theta_agrees_with_method_of_lines_and_boundary_value():
    heat = HeatEquation2D(lambda X, Y: np.zeros_like(X), n=(11, 11), boundary_value=2.0)
    mol = heat.solve(0.1, dt=0.5 * heat.max_stable_dt())
    cn = heat.solve_theta(0.1, dt=1e-3, theta=0.5)
    assert np.allclose(mol.final, cn.final, atol=1e-3)
    assert np.all(cn.final[0, :] == 2.0) and np.all(cn.final[:, -1] == 2.0)
    assert heat.max_stable_dt("ftcs") == pytest.approx(heat.dx**2 / 4)
    assert heat.max_stable_dt("crank_nicolson") == np.inf
    with pytest.raises(ValueError):
        heat.solve_theta(0.1, dt=1e-3, theta=-0.1)

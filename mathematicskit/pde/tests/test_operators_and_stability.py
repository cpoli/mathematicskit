"""Tests for mathematicskit.pde.utils.operators and mathematicskit.pde.systems.stability."""

import numpy as np
import pytest

from mathematicskit.pde import (
    RK4_IMAG_AXIS_LIMIT,
    RK4_REAL_AXIS_LIMIT,
    amplification_factor,
    check_cfl,
    check_diffusion_stability,
    courant_number,
    laplacian_1d,
    laplacian_2d,
    max_amplification,
    uniform_grid,
)


def _rk4_stability_function(z):
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def test_uniform_grid_spacing_matches_points():
    x, dx = uniform_grid(-1.0, 2.0, 7)
    assert np.allclose(np.diff(x), dx)
    xp, dxp = uniform_grid(0.0, 2 * np.pi, 8, periodic=True)
    assert dxp == pytest.approx(2 * np.pi / 8)
    assert xp[-1] == pytest.approx(2 * np.pi - dxp)


def test_uniform_grid_rejects_too_few_points():
    with pytest.raises(ValueError):
        uniform_grid(0.0, 1.0, 1)


def test_laplacian_1d_dirichlet_exact_on_quadratic_interior():
    n, dx = 9, 0.1
    x = dx * np.arange(1, n + 1)
    u = x**2
    lap = laplacian_1d(n, dx) @ u
    assert np.allclose(lap[1:-1], 2.0)  # away from the dropped boundary terms


def test_laplacian_1d_dirichlet_eigenvalues_match_closed_form():
    n, dx = 12, 1.0 / 13
    eigs = np.sort(np.linalg.eigvalsh(laplacian_1d(n, dx).toarray()))
    k = np.arange(1, n + 1)
    expected = np.sort(-4.0 / dx**2 * np.sin(k * np.pi / (2 * (n + 1))) ** 2)
    assert np.allclose(eigs, expected)


def test_laplacian_1d_periodic_and_neumann_annihilate_constants():
    for bc in ("periodic", "neumann"):
        assert np.allclose(laplacian_1d(8, 0.5, bc=bc) @ np.ones(8), 0.0)


def test_laplacian_1d_periodic_exact_on_fourier_mode():
    n = 16
    x, dx = uniform_grid(0.0, 2 * np.pi, n, periodic=True)
    lap = laplacian_1d(n, dx, bc="periodic") @ np.sin(x)
    assert np.allclose(lap, -(2 - 2 * np.cos(dx)) / dx**2 * np.sin(x))


def test_laplacian_1d_unknown_bc_raises():
    with pytest.raises(ValueError):
        laplacian_1d(4, 1.0, bc="robin")


def test_laplacian_2d_is_kronecker_sum_and_symmetric():
    L = laplacian_2d(4, 3, 0.5, 0.25).toarray()
    assert np.allclose(L, L.T)
    Lx, Ly = laplacian_1d(4, 0.5).toarray(), laplacian_1d(3, 0.25).toarray()
    assert np.allclose(L, np.kron(Lx, np.eye(3)) + np.kron(np.eye(4), Ly))


def test_rk4_axis_limits_lie_on_stability_boundary():
    assert abs(_rk4_stability_function(-RK4_REAL_AXIS_LIMIT)) == pytest.approx(1.0, abs=1e-12)
    assert abs(_rk4_stability_function(1j * RK4_IMAG_AXIS_LIMIT)) == pytest.approx(1.0, abs=1e-12)
    assert RK4_REAL_AXIS_LIMIT == pytest.approx(2.785293563405282)


def test_courant_number_uses_absolute_speed():
    assert courant_number(-2.0, 0.1, 0.4) == pytest.approx(0.5)


@pytest.mark.parametrize("scheme", ["upwind", "lax_friedrichs", "lax_wendroff"])
def test_cfl_limit_matches_von_neumann_analysis(scheme):
    for nu in np.linspace(0.05, 1.5, 30):
        von_neumann_stable = max_amplification(scheme, nu) <= 1.0 + 1e-12
        assert check_cfl(1.0, nu, 1.0, scheme).stable == von_neumann_stable


def test_ftcs_advection_is_unconditionally_unstable():
    assert not check_cfl(1.0, 1e-3, 1.0, "ftcs").stable
    assert max_amplification("ftcs_advection", 1e-3) > 1.0


def test_check_cfl_unknown_scheme_raises():
    with pytest.raises(ValueError):
        check_cfl(1.0, 0.1, 0.1, "beam_warming")


def test_diffusion_limit_matches_von_neumann_analysis():
    for r in np.linspace(0.05, 1.0, 20):
        assert check_diffusion_stability(1.0, r, 1.0).stable == (max_amplification("ftcs_heat", r) <= 1.0 + 1e-12)


@pytest.mark.parametrize("scheme", ["crank_nicolson_heat", "backward_euler_heat"])
def test_implicit_heat_schemes_are_unconditionally_stable(scheme):
    for r in (0.1, 1.0, 10.0, 1e4):
        assert max_amplification(scheme, r) <= 1.0


def test_theta_method_diffusion_limit_formula():
    assert check_diffusion_stability(1.0, 1.0, 1.0, theta=0.25).limit == pytest.approx(1.0)
    assert check_diffusion_stability(1.0, 1.0, 1.0, ndim=2).limit == pytest.approx(0.25)
    result = check_diffusion_stability(1.0, 1e3, 0.01, theta=1.0)
    assert result.stable and result.scheme == "backward_euler"


def test_amplification_factor_of_upwind_at_unit_courant_is_exact_shift():
    xi = np.linspace(0.0, np.pi, 9)
    assert np.allclose(amplification_factor("upwind", 1.0, xi), np.exp(-1j * xi))


def test_amplification_factor_unknown_scheme_raises():
    with pytest.raises(ValueError):
        amplification_factor("leapfrog_heat", 0.5, 0.0)

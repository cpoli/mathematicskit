"""Tests for mathematicskit.pde.systems.poisson and mathematicskit.pde.systems.spectral."""

import numpy as np
import pytest

from mathematicskit.pde import (
    BurgersEquation1D,
    HeatEquation1D,
    chebyshev_differentiation_matrix,
    chebyshev_poisson_1d,
    chebyshev_poisson_2d,
    fourier_derivative,
    fourier_differentiation_matrix,
    solve_laplace_2d,
    solve_poisson_1d,
    solve_poisson_2d,
)


def _source(X, Y):
    return -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)


def _exact(X, Y):
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def _poisson_error(n, **kwargs):
    sol = solve_poisson_2d(_source, n=(n, n), **kwargs)
    X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    return np.max(np.abs(sol.u - _exact(X, Y))), sol


def test_poisson_1d_is_exact_for_quadratic_and_second_order_otherwise():
    sol = solve_poisson_1d(lambda x: 6 * x, a=0.0, b=2.0, n=9, ua=1.0, ub=9.0)  # u = x^3 + 1
    errors = [np.max(np.abs(s.u - (s.x**3 + 1))) for s in (sol, solve_poisson_1d(lambda x: 6 * x, 0.0, 2.0, 17, 1.0, 9.0))]
    assert errors[0] < 1e-12  # cubic: the central difference is exact
    e = [np.max(np.abs(solve_poisson_1d(lambda x: -np.sin(x), 0, np.pi, n).u - np.sin(np.linspace(0, np.pi, n)))) for n in (11, 21, 41)]
    assert np.log2(e[0] / e[1]) == pytest.approx(2.0, abs=0.1)
    assert np.log2(e[1] / e[2]) == pytest.approx(2.0, abs=0.1)


def test_poisson_2d_direct_is_second_order():
    e1, sol = _poisson_error(17)
    e2, _ = _poisson_error(33)
    assert np.log2(e1 / e2) == pytest.approx(2.0, abs=0.1)
    assert sol.residual_norm < 1e-8 and sol.solver_result is None


@pytest.mark.parametrize("method", ["cg", "jacobi", "gauss_seidel", "sor"])
def test_iterative_poisson_matches_direct(method):
    _, direct = _poisson_error(13)
    _, iterative = _poisson_error(13, method=method, tol=1e-10)
    assert iterative.solver_result.converged
    assert np.allclose(iterative.u, direct.u, atol=1e-8)


def test_relaxation_iteration_counts_are_ordered():
    counts = {m: _poisson_error(13, method=m)[1].solver_result.iterations for m in ("jacobi", "gauss_seidel", "sor", "cg")}
    assert counts["jacobi"] > counts["gauss_seidel"] > counts["sor"]
    assert counts["gauss_seidel"] == pytest.approx(counts["jacobi"] / 2, rel=0.1)  # Gauss-Seidel converges twice as fast
    assert counts["cg"] < counts["sor"]


def test_poisson_2d_rejects_bad_inputs():
    with pytest.raises(ValueError):
        solve_poisson_2d(0.0, n=(9, 9), method="multigrid")
    with pytest.raises(ValueError):
        solve_poisson_2d(np.zeros((3, 3)), n=(9, 9))


def test_laplace_reproduces_harmonic_boundary_data_and_maximum_principle():
    g = lambda X, Y: X**2 - Y**2
    sol = solve_laplace_2d(g, n=(21, 11), x_range=(-1, 1), y_range=(0, 1))
    X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    assert np.allclose(sol.u, g(X, Y), atol=1e-10)  # the 5-point stencil is exact on quadratics

    boundary = np.zeros((21, 21))
    boundary[:, -1] = 1.0  # hot lid
    lid = solve_laplace_2d(boundary, n=(21, 21))
    assert lid.u.min() >= 0.0 and lid.u.max() <= 1.0
    assert lid.u[10, 10] == pytest.approx(0.25, abs=0.01)  # symmetry: the four-lid average at the center


def test_fourier_derivative_has_spectral_accuracy():
    f = lambda x: np.exp(np.sin(x))
    df = lambda x: np.cos(x) * np.exp(np.sin(x))
    errors = []
    for n in (8, 16, 32):
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        errors.append(np.max(np.abs(fourier_derivative(f(x)) - df(x))))
    assert errors[-1] < 1e-12
    assert errors[1] < errors[0] * 1e-3


def test_fourier_derivative_order_two_and_length():
    L = 3.0
    x = np.linspace(0, L, 24, endpoint=False)
    u = np.cos(2 * np.pi * 2 * x / L)
    assert np.allclose(fourier_derivative(u, length=L, order=2), -((4 * np.pi / L) ** 2) * u)
    assert np.allclose(fourier_derivative(np.ones(8)), 0.0)


@pytest.mark.parametrize("order", [1, 2])
def test_fourier_matrix_agrees_with_fft(order):
    n, L = 16, 5.0
    x = np.linspace(0, L, n, endpoint=False)
    u = np.exp(np.cos(2 * np.pi * x / L))
    D = fourier_differentiation_matrix(n, L, order=order)
    assert np.allclose(D @ u, fourier_derivative(u, length=L, order=order), atol=1e-10)


def test_fourier_matrix_rejects_bad_arguments():
    with pytest.raises(ValueError):
        fourier_differentiation_matrix(15)
    with pytest.raises(ValueError):
        fourier_differentiation_matrix(16, order=3)


def test_chebyshev_matrix_is_exact_for_polynomials_on_any_interval():
    D, x = chebyshev_differentiation_matrix(9, a=1.0, b=4.0)
    assert x[0] == pytest.approx(1.0) and x[-1] == pytest.approx(4.0)
    assert np.allclose(D @ x**8, 8 * x**7, rtol=1e-9)
    assert np.allclose(D @ np.ones(9), 0.0, atol=1e-12)


def test_chebyshev_poisson_1d_converges_geometrically():
    exact = lambda x: np.exp(x) * np.sin(3 * x)
    f = lambda x: np.exp(x) * (-8 * np.sin(3 * x) + 6 * np.cos(3 * x))
    errors = [np.max(np.abs((s := chebyshev_poisson_1d(f, n=n, ua=exact(-1.0), ub=exact(1.0))).u - exact(s.x))) for n in (8, 12, 16, 24)]
    assert errors[-1] < 1e-12
    assert errors[1] < 1e-2 * errors[0]
    assert chebyshev_poisson_1d(2.0, n=6).u == pytest.approx(chebyshev_poisson_1d(2.0, n=6).x ** 2 - 1)


def test_chebyshev_beats_finite_differences_at_equal_points():
    n = 17
    fd = solve_poisson_1d(lambda x: -(np.pi**2) * np.sin(np.pi * x), -1, 1, n)
    cheb = chebyshev_poisson_1d(lambda x: -(np.pi**2) * np.sin(np.pi * x), n=n)
    fd_err = np.max(np.abs(fd.u - np.sin(np.pi * fd.x)))
    cheb_err = np.max(np.abs(cheb.u - np.sin(np.pi * cheb.x)))
    assert cheb_err < 1e-6 * fd_err


def test_chebyshev_poisson_2d_is_spectrally_accurate():
    sol = chebyshev_poisson_2d(_source, n=20)
    X, Y = np.meshgrid(sol.x, sol.y, indexing="ij")
    assert np.max(np.abs(sol.u - _exact(X, Y))) < 1e-9
    assert np.all(sol.u[0, :] == 0.0) and np.all(sol.u[:, -1] == 0.0)
    assert np.allclose(chebyshev_poisson_2d(0.0, n=8).u, 0.0)


def test_burgers_conserves_mean_and_converges_with_resolution():
    u0 = lambda x: 0.5 + np.sin(x)
    finals = {n: BurgersEquation1D(u0, n=n, nu=0.1).solve(1.0, dt=1e-3, save_every=1000).final for n in (32, 64, 128)}
    assert finals[32].mean() == pytest.approx(0.5, abs=1e-12)
    e32 = np.max(np.abs(finals[32] - finals[128][::4]))
    e64 = np.max(np.abs(finals[64] - finals[128][::2]))
    assert e64 < 1e-3 * e32  # doubling n gains three digits: spectral, not algebraic, convergence


def test_burgers_small_amplitude_limit_is_the_heat_equation():
    eps, nu, t = 1e-6, 0.2, 1.0
    sol = BurgersEquation1D(lambda x: eps * np.sin(x), n=16, nu=nu).solve(t, dt=1e-3)
    assert np.allclose(sol.final / eps, np.exp(-nu * t) * np.sin(sol.x), atol=1e-5)
    heat = HeatEquation1D(lambda x: np.sin(x), length=2 * np.pi, n=16, alpha=nu, bc="periodic")
    assert heat.solve(t, dt=1e-3).final == pytest.approx(sol.final / eps, abs=5e-3)


def test_burgers_max_stable_dt_is_respected():
    burgers = BurgersEquation1D(lambda x: np.sin(x), n=32, nu=0.05)
    dt_max = burgers.max_stable_dt()
    assert 0 < dt_max < np.inf
    assert burgers.max_stable_dt("dopri5") == np.inf
    stable = burgers.solve(2.0, dt=0.9 * dt_max)
    assert stable.extra["stability"].stable and np.max(np.abs(stable.final)) < 1.0
    unstable = burgers.solve(2.0, dt=4.0 * dt_max)
    assert not unstable.extra["stability"].stable and not np.max(np.abs(unstable.final)) < 10.0

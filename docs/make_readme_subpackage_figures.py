"""Regenerate the README's per-subpackage teaser figures.

    python docs/make_readme_subpackage_figures.py             # all subpackages
    python docs/make_readme_subpackage_figures.py pde linalg  # just these

Writes docs/source/_static/images/readme_<subpackage>.png, three panels each,
at the same size as readme_hero.png (see make_readme_figure.py). README.md
embeds them by their raw.githubusercontent.com URLs so they also render on PyPI.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

OUT_DIR = Path(__file__).parent / "source" / "_static" / "images"


def numerical_analysis(axes):
    from mathematicskit.numerical_analysis import LagrangeInterpolant, bernstein_polynomial, runge_function

    ax1, ax2, ax3 = axes
    x_fine = np.linspace(-1.0, 1.0, 600)
    ax1.plot(x_fine, runge_function(x_fine), color="black", lw=2, label=r"$1/(1+25x^2)$")
    for n, color in ((6, "goldenrod"), (11, "darkorange"), (16, "firebrick")):
        nodes = np.linspace(-1.0, 1.0, n)
        ax1.plot(x_fine, LagrangeInterpolant(nodes, runge_function(nodes)).evaluate(x_fine), color=color, label=f"degree {n - 1}")
    ax1.set_ylim(-2.0, 2.0)
    ax1.set_title("Runge's phenomenon: more nodes, worse fit")
    ax1.legend(fontsize=8, loc="upper center")

    for n, color in ((11, "darkorange"), (21, "firebrick")):
        nodes = np.cos(np.pi * np.arange(n) / (n - 1))
        ax2.plot(x_fine, LagrangeInterpolant(nodes, runge_function(nodes)).evaluate(x_fine), color=color, label=f"degree {n - 1}")
        ax2.plot(nodes, runge_function(nodes), "o", ms=3, color=color)
    ax2.plot(x_fine, runge_function(x_fine), "k--", lw=1, label=r"$1/(1+25x^2)$")
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_title("Chebyshev nodes cure it")
    ax2.legend(fontsize=8, loc="upper center")

    f = lambda t: np.abs(t - 0.5)  # noqa: E731
    t = np.linspace(0.0, 1.0, 501)
    ax3.plot(t, f(t), "k", lw=2, label="$|x - 1/2|$")
    for n in (4, 16, 64, 256):
        ax3.plot(t, bernstein_polynomial(f, n, t), label=f"$B_{{{n}}}f$")
    ax3.set_title("Bernstein polynomials (Weierstrass theorem)")
    ax3.legend(fontsize=8, loc="upper center")


def linalg(axes):
    from mathematicskit.linalg import GMRES, ConjugateGradient, gershgorin_discs, random_spd_matrix, svd_decompose
    from mathematicskit.linalg.visualizers.plots import plot_residual_history

    ax1, ax2, ax3 = axes
    rng = np.random.default_rng(3)
    y, x = np.mgrid[-1:1:60j, -1:1:80j]
    a = np.exp(-(x**2 + 2 * y**2) * 3) + 0.5 * np.cos(4 * x * y) + 0.02 * rng.normal(size=x.shape)
    res = svd_decompose(a)
    tiles = [a] + [res.U[:, :k] @ np.diag(res.S[:k]) @ res.Vt[:k] for k in (1, 2, 5)]
    for tile, (x0, y0), label in zip(tiles, ((0, 64), (84, 64), (0, 0), (84, 0)), ("$A$", "rank 1", "rank 2", "rank 5"), strict=False):
        ax1.imshow(tile, cmap="viridis", vmin=a.min(), vmax=a.max(), extent=(x0, x0 + 80, y0, y0 + 60))
        ax1.text(x0 + 3, y0 + 55, label, color="white", va="top")
    ax1.set_xlim(0, 164)
    ax1.set_ylim(0, 124)
    ax1.axis("off")
    ax1.set_title("SVD: best low-rank approximations")

    d = np.diag([-4.0, 1.0, 3.0 + 2.0j, 3.0 - 2.0j, 7.0])
    m = d + 0.6 * rng.normal(size=(5, 5))
    discs = gershgorin_discs(m)
    eigs = np.linalg.eigvals(m)
    for c, r in zip(discs.centers, discs.radii, strict=False):
        ax2.add_patch(plt.Circle((c.real, c.imag), r, alpha=0.2))
        ax2.add_patch(plt.Circle((c.real, c.imag), r, fill=False))
    ax2.plot(eigs.real, eigs.imag, "k*", ms=10, label="eigenvalues")
    ax2.set_aspect("equal")
    ax2.autoscale_view()
    ax2.set_title("Gershgorin discs trap every eigenvalue")
    ax2.legend(fontsize=8, loc="lower left")

    b = np.ones(20)
    plot_residual_history(ConjugateGradient(tol=1e-10).solve(random_spd_matrix(20, seed=4), b), ax=ax3, label="conjugate gradient")
    a_gen = np.random.default_rng(5).uniform(-1, 1, size=(20, 20)) + 20.0 * np.eye(20)
    plot_residual_history(GMRES(tol=1e-10).solve(a_gen, b), ax=ax3, label="GMRES")
    ax3.set_title("Krylov solvers: residual per iteration")
    ax3.legend()


def calculus(axes):
    from mathematicskit.calculus import AdaptiveQuadrature, RiemannSum
    from mathematicskit.calculus.systems.taylor_series import evaluate_series, maclaurin_coefficients

    ax1, ax2, ax3 = axes
    n, h = 8, 1.0 / 8
    grid = np.linspace(0.0, 1.0, 200)
    left = h * np.arange(n)
    ax1.plot(grid, np.exp(grid), "k", lw=2)
    ax1.bar(left, np.exp(left + 0.5 * h), width=h, align="edge", alpha=0.4, edgecolor="k")
    value = RiemannSum(n, "midpoint").integrate(np.exp, 0.0, 1.0).value
    ax1.set_title(f"Midpoint Riemann sum: {value:.4f} vs. $e - 1$")

    coeffs = maclaurin_coefficients("sin", order=15)
    x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    ax2.plot(x, np.sin(x), "k", lw=2, label=r"$\sin x$")
    for degree in (1, 3, 5, 9, 15):
        ax2.plot(x, evaluate_series(coeffs[: degree + 1], x), "--", label=f"degree {degree}")
    ax2.set_ylim(-2, 2)
    ax2.set_title("Taylor polynomials of sin about 0")
    ax2.legend(fontsize=7, loc="lower left", ncol=2)

    peaked = lambda t: 1.0 / (1.0 + 1000.0 * (t - 0.3) ** 2)  # noqa: E731
    samples = []
    AdaptiveQuadrature(tol=1e-10).integrate(lambda t: samples.append(t) or peaked(t), 0.0, 1.0)
    fine = np.linspace(0.0, 1.0, 1000)
    ax3.plot(fine, peaked(fine), "k")
    ax3.plot(samples, peaked(np.asarray(samples)), "|", color="C0", ms=10)
    ax3.set_title("Adaptive quadrature samples where it matters")


def ode_dynamics(axes):
    from mathematicskit.ode_dynamics.systems.chaotic_flows import LorenzSystem
    from mathematicskit.ode_dynamics.systems.phase_portrait import Nonlinear2D, vector_field_grid
    from mathematicskit.ode_dynamics.systems.poincare import DuffingOscillator, stroboscopic_poincare_section
    from mathematicskit.ode_dynamics.visualizers.plots import plot_phase_portrait, plot_poincare_points, plot_vector_field

    ax1, ax2, ax3 = axes

    def pendulum(theta, omega):
        return omega, -np.sin(theta)

    plot_vector_field(*vector_field_grid(pendulum, (-3.5, 3.5), (-2.5, 2.5), n=20), ax=ax1)
    trajectories = [
        Nonlinear2D(state, f=pendulum).integrate((0.0, 10.0), dt=1e-3, method="rk4") for state in [(0.5, 0.0), (2.0, 0.0), (3.0, 0.0), (0.0, 2.2), (0.0, -2.2)]
    ]
    plot_phase_portrait(trajectories, ax=ax1, color="firebrick", lw=1.2)
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_title("Pendulum phase portrait")

    lorenz = LorenzSystem([1.0, 1.0, 1.0]).integrate((0.0, 40.0), dt=1e-3, method="rk4")
    ax2.plot(lorenz.y[2000:, 0], lorenz.y[2000:, 2], lw=0.4, color="tab:blue")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("The Lorenz attractor")

    duffing = DuffingOscillator([1.0, 0.0], delta=0.3, alpha=-1.0, beta=1.0, gamma=0.5, omega=1.2)
    xs, ys = stroboscopic_poincare_section(duffing, n_periods=4000, n_transient_periods=100, dt=1e-2)
    plot_poincare_points(xs, ys, ax=ax3)
    ax3.set_title("Poincaré section: the Duffing strange attractor")


def pde(axes):
    from mathematicskit.pde import BurgersConservationLaw1D, HeatEquation2D, WaveEquation1D
    from mathematicskit.pde.visualizers import plot_field_2d, plot_snapshots, plot_spacetime

    ax1, ax2, ax3 = axes
    heat = HeatEquation2D(lambda X, Y: np.where((np.abs(X - 0.5) < 0.2) & (np.abs(Y - 0.5) < 0.2), 1.0, 0.0), n=(41, 41))
    sol = heat.solve(0.01, dt=0.9 * heat.max_stable_dt("rk4"), save_every=40)
    plot_field_2d(sol, ax=ax1, cmap="inferno")
    ax1.set_title("2D heat equation: a hot square cools")

    string = WaveEquation1D(lambda x: np.maximum(0.0, 1.0 - np.abs(x - 0.3) / 0.1), n=201, c=1.0)
    plot_spacetime(string.solve(2.0, dt=string.dx, method="leapfrog", save_every=4), ax=ax2, cmap="RdBu_r")
    ax2.set_title("Wave equation: a plucked string, u(x, t)")

    burgers = BurgersConservationLaw1D(lambda x: 0.5 + np.sin(2 * np.pi * x), n=400, bc="periodic")
    plot_snapshots(burgers.solve(0.6, dt=0.5 * burgers.dx / 1.5, save_every=40), n_snapshots=5, ax=ax3)
    ax3.set_title("Burgers' equation: a shock forms (Godunov)")


def fractals_chaos(axes):
    from mathematicskit.fractals_chaos import BarnsleyFern, ElementaryCA, julia_set
    from mathematicskit.fractals_chaos.visualizers.plots import plot_ca_spacetime, plot_escape_time, plot_ifs_points

    ax1, ax2, ax3 = axes
    plot_ifs_points(BarnsleyFern().generate(80000, seed=0), ax=ax1, color="darkgreen")
    ax1.set_title("Barnsley's fern")

    rabbit = julia_set(c=-0.123 + 0.745j, extent=(-1.6, 1.6, -1.6, 1.6), resolution=500, max_iter=200)
    plot_escape_time(rabbit, ax=ax2, cmap="magma")
    # As in the hero figure: compress the escape counts and paint the filled Julia set black.
    image = ax2.images[0]
    image.set_data(np.ma.masked_equal(rabbit.iterations, rabbit.max_iter).astype(float) ** 0.3)
    image.autoscale()
    image.cmap.set_bad("black")
    ax2.set_title("Julia set: Douady's rabbit")

    plot_ca_spacetime(ElementaryCA(rule=30, width=161).run(80), ax=ax3)
    ax3.set_title("Wolfram's rule 30")


def optimization(axes):
    from mathematicskit.optimization import (
        BFGS,
        GradientDescent,
        NelderMead,
        NonlinearConjugateGradient,
        linear_program,
        quadratic_bowl,
        quadratic_bowl_grad,
        rosenbrock,
        rosenbrock_grad,
    )
    from mathematicskit.optimization.utils.comparison import compare_optimizers
    from mathematicskit.optimization.visualizers.plots import plot_contour_path, plot_convergence_comparison

    ax1, ax2, ax3 = axes
    x0 = [-1.2, 1.0]
    ranges = {"x_range": (-2.0, 2.0), "y_range": (-1.0, 3.0)}
    plot_contour_path(rosenbrock, NelderMead(tol=1e-8, max_iter=5000).minimize(rosenbrock, None, x0), ax=ax1, label="Nelder-Mead", **ranges)
    plot_contour_path(rosenbrock, BFGS(tol=1e-8).minimize(rosenbrock, rosenbrock_grad, x0), ax=ax1, label="BFGS", **ranges)
    ax1.set_title("Rosenbrock's banana: two optimizers' paths")
    ax1.legend(fontsize=8)

    results = compare_optimizers(
        {
            "gradient descent": GradientDescent(alpha=0.03, tol=1e-8, max_iter=2000),
            "conjugate gradient": NonlinearConjugateGradient(tol=1e-8, max_iter=2000),
        },
        quadratic_bowl,
        quadratic_bowl_grad,
        [5.0, -3.0],
    )
    plot_convergence_comparison(results, quadratic_bowl, f_star=0.0, ax=ax2)
    ax2.set_title("Conjugate gradient vs. gradient descent")

    c = np.array([3.0, 5.0])
    a_ub = np.array([[1.0, 2.0], [2.0, 1.0]])
    b_ub = np.array([40.0, 30.0])
    optimum = linear_program(-c, a_ub=a_ub, b_ub=b_ub).x
    polygon = np.array([[0.0, 0.0], [15.0, 0.0], [20.0 / 3.0, 50.0 / 3.0], [0.0, 20.0]])
    ax3.fill(*polygon.T, color="0.9", label="feasible region")
    xs = np.linspace(0.0, 25.0, 2)
    for level in (50.0, 100.0, c @ optimum):
        ax3.plot(xs, (level - 3.0 * xs) / 5.0, ":", color="tab:green", lw=1)
    ax3.plot(*polygon.T, "ko", ms=5)
    ax3.plot(*optimum, "r*", ms=15, label="optimum vertex")
    ax3.set_xlim(-1.0, 25.0)
    ax3.set_ylim(-1.0, 25.0)
    ax3.set_title("Linear programming: optimum at a vertex")
    ax3.legend(fontsize=8)


def probability(axes):
    from mathematicskit.probability import Exponential, brownian_motion
    from mathematicskit.probability.systems.limit_theorems import central_limit_theorem_sample_means
    from mathematicskit.probability.systems.monte_carlo import monte_carlo_integrate
    from mathematicskit.probability.visualizers.plots import plot_clt_histogram

    ax1, ax2, ax3 = axes
    plot_clt_histogram(central_limit_theorem_sample_means(Exponential(rate=1.0), n=200, n_trials=5000, seed=0), ax=ax1)
    ax1.set_title("Central limit theorem: means of Exponential(1)")

    walk = brownian_motion(n_paths=30, n_steps=1000, t_max=1.0, seed=0)
    ax2.plot(walk.times, walk.paths.T, lw=0.8)
    ax2.plot(walk.times, 2 * np.sqrt(walk.times), "k--", walk.times, -2 * np.sqrt(walk.times), "k--")
    ax2.set_title(r"Brownian motion and its $\pm 2\sqrt{t}$ envelope")

    ns = np.unique(np.logspace(1, 5, 25).astype(int))
    estimates = [monte_carlo_integrate(np.exp, 0.0, 1.0, n=int(n), seed=0) for n in ns]
    values = np.array([r.estimate for r in estimates])
    errors = np.array([r.std_error for r in estimates])
    ax3.semilogx(ns, values, "o-", label="estimate")
    ax3.fill_between(ns, values - 2 * errors, values + 2 * errors, alpha=0.3, label=r"$\pm 2$ std. errors")
    ax3.axhline(np.e - 1.0, color="k", ls="--", label="exact $e - 1$")
    ax3.set_xlabel("samples $n$")
    ax3.set_title(r"Monte Carlo: $\int_0^1 e^x\,dx$")
    ax3.legend(fontsize=8)


def statistics(axes):
    from mathematicskit.statistics import linear_regression, pearson_correlation
    from mathematicskit.statistics.visualizers.plots import plot_boxplot, plot_regression_fit, plot_residuals

    ax1, ax2, ax3 = axes
    rng = np.random.default_rng(0)
    parent = rng.normal(loc=68.0, scale=1.8, size=400)
    child = 68.0 + 0.65 * (parent - 68.0) + rng.normal(scale=2.0, size=400)
    fit = linear_regression(parent, child)
    plot_regression_fit(parent, child, fit, ax=ax1)
    ax1.set_title(f"Galton's heights: r = {pearson_correlation(parent, child).coefficient:.2f}")

    plot_residuals(fit, ax=ax2)
    ax2.set_title("OLS residual diagnostics")

    groups = np.column_stack([rng.normal(mu, 1.0, 60) for mu in (5.0, 5.8, 6.6)])
    plot_boxplot(groups, ax=ax3)
    ax3.set_title("Comparing groups: box plots (one-way ANOVA)")


def number_theory(axes):
    from mathematicskit.number_theory import continued_fraction_expansion, sieve_of_eratosthenes, sum_of_two_squares
    from mathematicskit.number_theory.visualizers.plots import plot_convergent_errors, plot_prime_counting

    ax1, ax2, ax3 = axes
    plot_prime_counting(2000, ax=ax1)
    ax1.set_title("Prime-counting function vs. the PNT")

    plot_convergent_errors(continued_fraction_expansion(np.pi, max_terms=8), x=np.pi, ax=ax2)
    ax2.set_xscale("log")
    ax2.set_title(r"Continued-fraction convergents of $\pi$")

    ones = [int(p) for p in sieve_of_eratosthenes(10000) if p % 4 == 1]
    ab = np.array([sum_of_two_squares(p) for p in ones])
    ax3.plot(ab[:, 0], ab[:, 1], ".", ms=3, color="tab:blue")
    ax3.plot(ab[:, 1], ab[:, 0], ".", ms=3, color="tab:orange")
    ax3.set_aspect("equal")
    ax3.set_title(r"Fermat: primes $p = a^2 + b^2$, $p \equiv 1 \ (\mathrm{mod}\ 4)$")


def combinatorics(axes):
    from mathematicskit.combinatorics import pascals_triangle
    from mathematicskit.combinatorics.visualizers.plots import plot_partition_counts, plot_pascals_triangle

    ax1, ax2, ax3 = axes
    plot_pascals_triangle(pascals_triangle(10), ax=ax1)
    ax1.set_title("Pascal's triangle")

    rows = pascals_triangle(64)
    parity = np.full((len(rows), len(rows)), np.nan)
    for n, row in enumerate(rows):
        parity[n, : len(row)] = np.asarray(row, dtype=object) % 2
    ax2.imshow(parity, cmap="Greys", interpolation="nearest")
    ax2.axis("off")
    ax2.set_title("Pascal's triangle mod 2: Sierpiński's triangle")

    plot_partition_counts(30, ax=ax3)
    ax3.set_title("The partition function p(n)")


def complex_analysis(axes):
    from mathematicskit.complex_analysis import circle_contour, domain_coloring, joukowski_map
    from mathematicskit.complex_analysis.visualizers import plot_contour, plot_domain_coloring

    ax1, ax2, ax3 = axes
    plot_domain_coloring(
        domain_coloring(lambda z: (z - 1) ** 2 * (z + 1) / (z**2 + 1), (-2, 2), (-2, 2), resolution=400),
        ax=ax1,
        title=r"Domain coloring of $(z-1)^2(z+1)/(z^2+1)$",
    )

    for c in (0.0, -0.1, -0.1 + 0.15j):
        z = circle_contour(c, abs(1 - c)).points(600)
        w = joukowski_map(z)
        (line,) = ax2.plot(w.real, w.imag, lw=2)
        ax2.plot(z.real, z.imag, ":", color=line.get_color(), lw=1)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_title(r"Joukowski $w = z + 1/z$: circles (dotted) to airfoils")

    poles = [1, -2, 3j]
    for radius in (0.5, 1.5, 2.5, 3.5):
        plot_contour(circle_contour(0.0, radius), marked_points=poles, ax=ax3)
    ax3.set_title("Residue theorem: contours around three poles")


def graph_theory(axes):
    from mathematicskit.graph_theory import Graph, astar_shortest_path, backtracking_coloring, grid_graph, spectral_analysis
    from mathematicskit.graph_theory.utils.generators import complete_graph
    from mathematicskit.graph_theory.visualizers.plots import plot_spectral_bipartition

    ax1, ax2, ax3 = axes
    k = 5
    barbell = Graph(2 * k)
    for u, v, w in complete_graph(k).edges():
        barbell.add_edge(u, v, w)
        barbell.add_edge(u + k, v + k, w)
    barbell.add_edge(k - 1, k)
    plot_spectral_bipartition(barbell, spectral_analysis(barbell), ax=ax1)
    ax1.set_title("Fiedler's spectral bipartition")

    rows = cols = 30
    blocked = [(r, 15) for r in range(0, 24)] + [(8, c) for c in range(3, 15)]
    g, pos = grid_graph(rows, cols, blocked=blocked)
    source, target = 2 * cols + 2, (rows - 3) * cols + (cols - 3)
    path = astar_shortest_path(g, source, target, lambda v: float(np.abs(pos[v] - pos[target]).sum())).path
    ax2.plot(*np.array([(c, r) for r, c in blocked]).T, "ks", ms=4)
    ax2.plot(*pos[path].T, "-", color="tab:red", lw=2.5)
    ax2.plot(*pos[[source, target]].T, "o", color="tab:green", ms=9)
    ax2.set_aspect("equal")
    ax2.invert_yaxis()
    ax2.set_title("A* search around obstacles")

    rng = np.random.default_rng(3)
    ring = [(0.5 + 0.2 * np.cos(t), 0.5 + 0.2 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    outer = rng.uniform(0.0, 1.0, size=(60, 2))
    capitals = np.vstack([[0.5, 0.5], ring, outer[np.hypot(*(outer - 0.5).T) > 0.35][:14]])
    size = 300
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    labels = np.argmin((xx[..., None] - capitals[:, 0]) ** 2 + (yy[..., None] - capitals[:, 1]) ** 2, axis=-1)
    countries = Graph(len(capitals))
    for a, b in zip(
        np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()]),
        np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()]),
        strict=False,
    ):
        if a != b:
            countries.add_edge(int(a), int(b))
    coloring = backtracking_coloring(countries)
    palette = ListedColormap(["#e15759", "#4e79a7", "#59a14f", "#edc948"])
    ax3.imshow(np.vectorize(coloring.coloring.get)(labels), cmap=palette, vmin=0, vmax=3, origin="lower", interpolation="nearest")
    border = (np.diff(labels, axis=0, prepend=labels[:1]) != 0) | (np.diff(labels, axis=1, prepend=labels[:, :1]) != 0)
    ax3.imshow(np.ma.masked_where(~border, border), cmap=ListedColormap(["black"]), origin="lower", interpolation="nearest")
    ax3.axis("off")
    ax3.set_title(f"Map coloring with {coloring.num_colors} colors")


def abstract_algebra(axes):
    from mathematicskit.abstract_algebra import GF, DihedralGroup, PermutationGroup, find_irreducible_polynomial
    from mathematicskit.abstract_algebra.visualizers.plots import plot_cayley_table

    ax1, ax2, ax3 = axes
    plot_cayley_table(DihedralGroup(4), ax=ax1)
    ax1.set_title("Cayley table of $D_4$, the square's symmetries")

    plot_cayley_table(PermutationGroup(4), ax=ax2)
    ax2.set_title("Cayley table of $S_4$ (order 24)")

    field = GF(2, 4, irreducible=find_irreducible_polynomial(2, 4))
    elements = list(field.elements())
    index = {tuple(e.coeffs): i for i, e in enumerate(elements)}
    table = np.array([[index[tuple(field.multiply(a, b).coeffs)] for b in elements] for a in elements])
    ax3.imshow(table, cmap="tab20")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Multiplication table of the field GF(16)")


def geometry(axes):
    from mathematicskit.geometry import bezier_curve, de_casteljau, delaunay_triangulation, voronoi_diagram
    from mathematicskit.geometry.visualizers.plots import plot_triangulation, plot_voronoi

    ax1, ax2, ax3 = axes
    rng = np.random.default_rng(0)
    sites = rng.uniform(0, 10, size=(15, 2))
    voronoi_diagram(sites)
    xs = np.linspace(0, 10, 300)
    X, Y = np.meshgrid(xs, xs)
    nearest = np.argmin(np.hypot(X[..., None] - sites[:, 0], Y[..., None] - sites[:, 1]), axis=-1)
    ax1.imshow(nearest, origin="lower", extent=(0, 10, 0, 10), cmap="tab20", alpha=0.5)
    plot_voronoi(sites, ax=ax1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect("equal")
    ax1.set_title("Voronoi diagram")

    points = np.random.default_rng(0).uniform(0, 10, size=(15, 2))
    tri = delaunay_triangulation(points)
    plot_triangulation(tri, ax=ax2)
    for simplex in tri.simplices:
        a, b, c = points[simplex]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        sq = [p @ p for p in (a, b, c)]
        center = np.array(
            [
                (sq[0] * (b[1] - c[1]) + sq[1] * (c[1] - a[1]) + sq[2] * (a[1] - b[1])) / d,
                (sq[0] * (c[0] - b[0]) + sq[1] * (a[0] - c[0]) + sq[2] * (b[0] - a[0])) / d,
            ]
        )
        ax2.add_patch(plt.Circle(center, np.linalg.norm(a - center), fill=False, color="tab:orange", lw=0.6, alpha=0.6))
    ax2.set_xlim(-2, 12)
    ax2.set_ylim(-2, 12)
    ax2.set_aspect("equal")
    ax2.set_title("Delaunay: every circumcircle is empty")

    control = np.array([[0.0, 0.0], [1.0, 3.0], [3.5, 3.5], [4.5, 0.5]])
    ax3.plot(*bezier_curve(control, np.linspace(0, 1, 200)).T, "k", lw=2)
    levels = de_casteljau(control, 0.4)
    for level, color in zip(levels, ("tab:gray", "tab:blue", "tab:green", "tab:red"), strict=False):
        ax3.plot(*level.T, "o-", color=color, ms=6)
    ax3.set_aspect("equal")
    ax3.set_title("Bézier curve by de Casteljau's algorithm")


def special_functions(axes):
    from mathematicskit.special_functions import bessel_first_kind, bessel_second_kind, fresnel_integrals, morlet_cwt
    from mathematicskit.special_functions.visualizers.plots import plot_scalogram

    ax1, ax2, ax3 = axes
    x = np.linspace(0.1, 15.0, 400)
    for nu in (0.0, 1.0, 2.0):
        ax1.plot(x, bessel_first_kind(nu, x), label=rf"$J_{int(nu)}(x)$")
    ax1.plot(x, bessel_second_kind(0.0, x), "--", label=r"$Y_0(x)$")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylim(-1.2, 1.1)
    ax1.set_title("Bessel functions")
    ax1.legend(fontsize=8, ncol=2)

    dt = 0.002
    t = np.arange(0.0, 4.0, dt)
    signal = np.sin(2 * np.pi * (5.0 * t + 5.0 * t**2)) + np.exp(-(((t - 3.0) / 0.05) ** 2)) * np.sin(2 * np.pi * 80.0 * t)
    plot_scalogram(morlet_cwt(signal, np.geomspace(0.005, 0.4, 120), dt=dt), ax=ax2)
    ax2.set_title("Morlet wavelet scalogram of a chirp")

    r = fresnel_integrals(np.linspace(-6, 6, 3000))
    ax3.plot(r.c, r.s, lw=1)
    ax3.set_aspect("equal")
    ax3.set_title("Fresnel integrals: the Cornu spiral")


def information_theory(axes):
    from mathematicskit.information_theory import bsc_transmit, convolutional_encode, hamming_decode, hamming_encode, huffman_code, viterbi_decode
    from mathematicskit.information_theory.visualizers import plot_code_tree, plot_information_diagram

    ax1, ax2, ax3 = axes
    plot_code_tree(huffman_code({"A": 0.35, "B": 0.17, "C": 0.17, "D": 0.16, "E": 0.15}), ax=ax1)
    ax1.set_title("A Huffman code tree")

    p = 0.1
    plot_information_diagram(0.5 * np.array([[1 - p, p], [p, 1 - p]]), ax=ax2)
    ax2.set_title("Mutual information across a BSC(0.1)")

    rng = np.random.default_rng(0)
    ps = np.logspace(-2.3, -0.9, 8)
    bits = rng.integers(0, 2, 40000)
    hamming = np.array([np.mean(hamming_decode(bsc_transmit(hamming_encode(bits), q, seed=rng)) != bits) for q in ps])
    viterbi = np.array([np.mean(viterbi_decode(bsc_transmit(convolutional_encode(bits), q, seed=rng)) != bits) for q in ps])
    ax3.loglog(ps, ps, "k--", label="uncoded")
    ax3.loglog(ps[hamming > 0], hamming[hamming > 0], "o-", label="Hamming (7,4)")
    ax3.loglog(ps[viterbi > 0], viterbi[viterbi > 0], "s-", label="(7,5) code, Viterbi")
    ax3.set_xlabel("crossover probability")
    ax3.set_ylabel("bit error rate")
    ax3.legend(fontsize=8)
    ax3.set_title("Error-correcting codes on a binary symmetric channel")


def topology(axes):
    from mathematicskit.topology import (
        critical_points,
        persistent_homology,
        torus,
        vietoris_rips_complex,
        vietoris_rips_filtration,
    )
    from mathematicskit.topology.visualizers import plot_complex, plot_persistence_diagram

    ax1, ax2, ax3 = axes
    fig = ax1.figure
    ax1.remove()
    ax1 = fig.add_subplot(1, 3, 1, projection="3d", computed_zorder=False)
    T = torus(16, 10)
    height = T.coordinates[:, 0] + 1e-3 * T.coordinates[:, 2]
    plot_complex(T, ax=ax1, face_values=[height[list(t)].mean() for t in T.simplices(2)], alpha=0.6, vertex_size=0)
    crit = critical_points(T, height)
    for vertices, color in ((crit.minima, "tab:blue"), (crit.saddles, "tab:red"), (crit.maxima, "gold")):
        ax1.scatter(*T.coordinates[vertices].T, s=60, color=color, edgecolors="k", depthshade=False, zorder=5)
    ax1.view_init(elev=25, azim=-90)
    ax1.set_box_aspect((1, 1, 1), zoom=1.3)
    ax1.set_axis_off()
    ax1.set_title("Critical points of a height function on a torus")

    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 60)
    points = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, 0.06, (60, 2))
    plot_complex(vietoris_rips_complex(points, 0.25), ax=ax2)
    ax2.set_title("Vietoris-Rips complex of a noisy circle")

    plot_persistence_diagram(persistent_homology(vietoris_rips_filtration(points, max_dim=2), max_dim=1), ax=ax3)
    ax3.set_title("Its persistence diagram: one long-lived loop")


FIGURES = {
    f.__name__: f
    for f in (
        numerical_analysis,
        linalg,
        calculus,
        ode_dynamics,
        pde,
        fractals_chaos,
        optimization,
        probability,
        statistics,
        number_theory,
        combinatorics,
        complex_analysis,
        graph_theory,
        abstract_algebra,
        geometry,
        special_functions,
        information_theory,
        topology,
    )
}


def main(names):
    for name in names or FIGURES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        FIGURES[name](axes)
        out = OUT_DIR / f"readme_{name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])

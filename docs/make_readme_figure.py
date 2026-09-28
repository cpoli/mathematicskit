"""Regenerate the README hero figure: python docs/make_readme_figure.py

Writes docs/source/_static/images/readme_hero.png, which README.md embeds by
its raw.githubusercontent.com URL so it also renders on PyPI.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mathematicskit as mk
from mathematicskit.fractals_chaos.visualizers import plot_escape_time
from mathematicskit.numerical_analysis.visualizers import plot_convergence_history
from mathematicskit.ode_dynamics.visualizers import plot_bifurcation_diagram

OUT = Path(__file__).parent / "source" / "_static" / "images" / "readme_hero.png"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

# Newton's quadratic convergence vs. bisection's linear convergence on x^2 - 2.
f, fprime = (lambda x: x**2 - 2.0), (lambda x: 2.0 * x)
for result in (
    mk.numerical_analysis.Bisection(f, 0.0, 3.0, tol=1e-14).solve(),
    mk.numerical_analysis.NewtonRaphson(f, fprime, x0=3.0, tol=1e-14).solve(),
):
    plot_convergence_history(result, root_exact=np.sqrt(2.0), ax=ax2)
ax2.set_title("Newton vs. bisection: the error, iterate by iterate")
ax2.legend()

escape = mk.fractals_chaos.mandelbrot_set(resolution=700, max_iter=200)
plot_escape_time(escape, ax=ax1, cmap="magma")
# Compress the escape counts (power 0.3) and paint the set itself black, for contrast.
image = ax1.images[0]
image.set_data(np.ma.masked_equal(escape.iterations, escape.max_iter).astype(float) ** 0.3)
image.autoscale()
image.cmap.set_bad("black")
ax1.set_title("The Mandelbrot set")

r_plot, x_plot = mk.ode_dynamics.bifurcation_diagram(np.linspace(2.8, 4.0, 1500), n_keep=150)
plot_bifurcation_diagram(r_plot, x_plot, ax=ax3)
ax3.set_title("Logistic map: the Feigenbaum route to chaos")

fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")

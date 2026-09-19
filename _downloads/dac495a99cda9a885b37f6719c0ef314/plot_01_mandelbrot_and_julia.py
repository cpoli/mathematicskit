r"""
Mandelbrot set and a companion Julia set
==============================================

Renders the Mandelbrot set alongside the Julia set for a parameter
``c`` picked from just outside the main cardioid, where the Julia set
is a connected, dendrite-like fractal.
"""

# %%
from mathematicskit.fractals_chaos import julia_set, mandelbrot_set
from mathematicskit.fractals_chaos.visualizers.plots import plot_escape_time

# %%
# Mandelbrot set
# ---------------------

mandelbrot = mandelbrot_set(resolution=300, max_iter=200)
plot_escape_time(mandelbrot)

# %%
# Julia set for c = -0.4 + 0.6i
# ------------------------------------

julia = julia_set(c=-0.4 + 0.6j, resolution=300, max_iter=200)
plot_escape_time(julia, cmap="magma")

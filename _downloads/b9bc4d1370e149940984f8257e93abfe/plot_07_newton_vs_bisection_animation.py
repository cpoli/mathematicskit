r"""
Animated: Newton's tangents against bisection's halving
=======================================================

Bisection keeps a bracket around the root and halves it every step, so
it gains one binary digit per iteration: linear convergence. Newton's
method slides down the tangent line at the current guess and, near a
simple root, doubles the number of correct digits per iteration:
quadratic convergence. This animation runs both on Newton's own cubic
:math:`f(x) = x^3 - 2x - 5`, one iteration per frame, with the error
of each growing in the bottom panel.
"""

# %%
import matplotlib.pyplot as plt

from mathematicskit.numerical_analysis import Bisection, NewtonRaphson
from mathematicskit.numerical_analysis.visualizers import animate_root_finding

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"


def f(x):
    return x**3 - 2.0 * x - 5.0


def fprime(x):
    return 3.0 * x**2 - 2.0


bisection = Bisection(f, 1.0, 3.0, tol=1e-12).solve()
newton = NewtonRaphson(f, fprime, x0=3.0, tol=1e-15).solve()
print(f"bisection: {bisection.iterations} iterations, Newton: {newton.iterations} iterations")

# %%
# One iteration per frame
# -----------------------
# The orange band is bisection's current bracket; the red segment is
# Newton's tangent from :math:`(x_{k-1}, f(x_{k-1}))` down to
# :math:`x_k`. Newton reaches machine precision while bisection is still
# a few digits in, then holds still as bisection keeps halving.

anim = animate_root_finding(f, [bisection, newton], x_range=(0.8, 3.2))
anim

# %%
plt.show()

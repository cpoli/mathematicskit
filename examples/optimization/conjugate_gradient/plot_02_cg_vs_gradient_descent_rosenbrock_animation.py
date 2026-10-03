r"""
Animated: conjugate gradient against gradient descent in Rosenbrock's valley
============================================================================

Rosenbrock's function :math:`(1 - x)^2 + 100(y - x^2)^2` has its minimum
at :math:`(1, 1)` at the bottom of a long, curved, flat-floored valley.
Steepest descent, even with a line search, zig-zags across the valley
because each new gradient is orthogonal to the last step. Nonlinear
conjugate gradient (Polak-Ribiere) mixes the previous direction into the
next, so it follows the valley floor. Frames are spaced geometrically in
the iteration count, so the title shows how long each method has been
walking.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.optimization import GradientDescentLineSearch, NonlinearConjugateGradient, rosenbrock, rosenbrock_grad
from mathematicskit.optimization.utils.comparison import compare_optimizers
from mathematicskit.optimization.visualizers import animate_optimizer_paths

# In Jupyter or JupyterLite, show animations as an HTML/JavaScript player
# (no ffmpeg needed).
plt.rcParams["animation.html"] = "jshtml"

optimizers = {
    "gradient descent (line search)": GradientDescentLineSearch(tol=1e-6, max_iter=20000),
    "conjugate gradient (Polak-Ribiere)": NonlinearConjugateGradient(tol=1e-6, max_iter=20000),
}
results = compare_optimizers(optimizers, rosenbrock, rosenbrock_grad, [-1.2, 1.0])
for name, result in results.items():
    print(f"{name}: {result.iterations} iterations, x = {np.round(result.x, 6)}")

# %%
# Both paths, frame by frame
# --------------------------
# Geometrically spaced contour levels show the valley floor, where the
# function is nearly flat.

anim = animate_optimizer_paths(rosenbrock, results, x_range=(-1.5, 1.5), y_range=(-0.5, 1.5), n_grid=150, levels=np.geomspace(1e-2, 1e3, 18), n_frames=50)
anim

# %%
plt.show()

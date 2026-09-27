r"""
Domain coloring: reading zeros, poles, and branch cuts from a phase portrait
============================================================================

Each point :math:`z` is colored by :math:`w = f(z)`: hue for the phase
:math:`\arg w`, brightness bands for every doubling of :math:`|w|`. A
zero is where all colors meet counter-clockwise, a pole where they meet
clockwise, and the order is the number of times each color appears
around the point. The technique was named by Frank Farris in the 1990s
and developed systematically by Elias Wegert.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import domain_coloring
from mathematicskit.complex_analysis.visualizers import plot_domain_coloring

functions = {
    r"$z$": lambda z: z,
    r"$(z - 1)^2 (z + 1) / (z^2 + 1)$": lambda z: (z - 1) ** 2 * (z + 1) / (z**2 + 1),
    r"$\sin z$": np.sin,
    r"$\sqrt{z}$ (branch cut on $(-\infty, 0]$)": np.sqrt,
    r"$e^{1/z}$ (essential singularity)": lambda z: np.exp(1 / z),
    r"$\tan z$": np.tan,
}
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
for ax, (title, f) in zip(axes.ravel(), functions.items(), strict=False):
    x_range = (-np.pi, np.pi) if "sin" in title or "tan" in title else (-2, 2)
    plot_domain_coloring(domain_coloring(f, x_range, (-2, 2), resolution=300), ax=ax, title=title)
fig.tight_layout()

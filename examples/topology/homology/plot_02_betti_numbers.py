r"""
Betti's connectivity numbers (1871)
===================================

Enrico Betti counted the independent closed curves, surfaces, and so
on, that do not bound in a space. In Poincaré's later algebra these are
the ranks :math:`\beta_k = \dim\ker\partial_k - \operatorname{rk}\partial_{k+1}`:
cycles minus boundaries. :math:`\beta_0` counts pieces, :math:`\beta_1`
independent loops, and :math:`\beta_2` enclosed cavities. The bars show
the bookkeeping for each space: chains :math:`f_k`, cycles, and
boundaries.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import SimplicialComplex, betti_numbers, circle, sphere, torus

theta = SimplicialComplex([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [4, 2]])
spaces = {
    "two points": SimplicialComplex([[0], [1]]),
    "circle": circle(6),
    "theta graph": theta,
    "sphere": sphere(2),
    "torus": torus(6, 4),
    "sphere + circle": sphere(2) | SimplicialComplex([[i + 10, (i + 1) % 4 + 10] for i in range(4)]),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, (name, K) in zip(axes.ravel(), spaces.items(), strict=True):
    ranks = [np.linalg.matrix_rank(K.boundary_matrix(k)) if K.boundary_matrix(k).size else 0 for k in range(K.dimension + 2)]
    f = np.array(K.f_vector)
    cycles = f - np.array(ranks[: len(f)])
    boundaries = np.array(ranks[1 : len(f) + 1])
    betti = betti_numbers(K)
    print(f"{name:>16}: f = {K.f_vector}, Betti numbers = {betti}")
    k = np.arange(len(f))
    ax.bar(k - 0.25, cycles, 0.25, label="cycles: dim ker ∂_k")
    ax.bar(k, boundaries, 0.25, label="boundaries: rk ∂_(k+1)")
    ax.bar(k + 0.25, betti, 0.25, label="β_k", color="k")
    for i, b in enumerate(betti):
        ax.annotate(str(b), (i + 0.25, b), ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(k, [f"k={i}" for i in k])
    ax.set_title(f"{name}: β = {betti}")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Betti numbers: cycles that are not boundaries")

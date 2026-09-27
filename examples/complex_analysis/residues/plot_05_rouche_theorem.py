r"""
Rouché's theorem: counting zeros by domination
==============================================

Eugène Rouché (1862): if :math:`|g| < |f|` on a closed contour, then
:math:`f` and :math:`f + g` have the same number of zeros inside it.
Picture :math:`f(\gamma)` as a person walking around a lamppost at 0
and :math:`g` as the leash to a dog at :math:`f + g`: a leash shorter
than the distance to the post means the dog circles it the same number
of times. Here :math:`z^5 + 3z + 1` has all five zeros in
:math:`|z| < 2` (dominated by :math:`z^5`) and exactly one in
:math:`|z| < 1` (dominated by :math:`3z`).
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.complex_analysis import argument_principle, circle_contour


def h(z):
    return z**5 + 3 * z + 1


roots = np.roots([1, 0, 0, 0, 3, 1])

# %%
# Two contours, two dominant terms
# --------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (R, f, name) in zip(axes, [(2.0, lambda z: z**5, "z^5"), (1.0, lambda z: 3 * z, "3z")], strict=False):
    contour = circle_contour(0.0, R)
    z = contour.points(2000)
    g = h(z) - f(z)
    print(f"|z| = {R}: min |{name}| = {np.min(np.abs(f(z))):.2f} > max |rest| = {np.max(np.abs(g)):.2f}")
    print(
        f"   zeros of {name}: {argument_principle(f, contour)}, zeros of z^5+3z+1: {argument_principle(h, contour)}, "
        f"numpy.roots inside: {int(np.sum(np.abs(roots) < R))}"
    )
    ax.plot(f(z).real, f(z).imag, lw=1, label=f"walker: {name}")
    ax.plot(h(z).real, h(z).imag, lw=1, label="dog: z^5 + 3z + 1")
    ax.plot(0, 0, "k*", ms=12, label="lamppost 0")
    ax.set_aspect("equal")
    ax.set_title(f"|z| = {R}")
    ax.legend(fontsize=8)
fig.suptitle("Rouché's theorem as dog walking")
fig.tight_layout()

plt.show()

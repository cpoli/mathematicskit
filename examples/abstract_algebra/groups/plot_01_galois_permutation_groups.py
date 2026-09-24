r"""
Galois's permutation groups: S_3 permuting the roots of x^3 - 2
======================================================================

Galois turned the question "is this equation solvable?" into a question
about a group of permutations of its roots. The roots of
:math:`x^3 - 2` are :math:`\sqrt[3]{2}`, :math:`\sqrt[3]{2}\,\omega` and
:math:`\sqrt[3]{2}\,\omega^2` (with :math:`\omega = e^{2\pi i/3}`).
Complex conjugation swaps the two non-real roots, multiplication by
:math:`\omega` cycles all three, and closing these two permutations
under composition gives Galois's group -- the full symmetric group
:math:`S_3`, which is non-abelian.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.abstract_algebra import PermutationGroup, group_properties

# %%
# The roots and the two generating permutations
# -----------------------------------------------------

roots = np.cbrt(2.0) * np.exp(2j * np.pi * np.arange(3) / 3)
print("roots of x^3 - 2:", np.round(roots, 4))
print("max |r^3 - 2|:", np.max(np.abs(roots**3 - 2)))

conjugation = (0, 2, 1)  # root 0 is real; roots 1 and 2 swap
rotation = (1, 2, 0)  # r_k -> r_{k+1}: multiply by omega
galois = PermutationGroup(3, generators=[conjugation, rotation])
print(f"closure of the 2 generators: {galois.order} permutations")
print(f"equals the full symmetric group S_3: {galois.order == PermutationGroup(3).order}")

# %%
# Non-commutativity -- the feature Galois's theory turns on
# -----------------------------------------------------------------

ab = galois.operate(conjugation, rotation)
ba = galois.operate(rotation, conjugation)
print(f"conj . rot = {ab},  rot . conj = {ba},  equal: {ab == ba}")
print(f"abelian: {group_properties(galois).is_abelian}")

# %%
# Every element of the Galois group, drawn as arrows between roots
# -----------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
circle = np.exp(1j * np.linspace(0, 2 * np.pi, 200)) * np.cbrt(2.0)
for ax, perm in zip(axes.flat, galois.elements):
    ax.plot(circle.real, circle.imag, color="0.85", lw=1)
    ax.scatter(roots.real, roots.imag, s=60, color="C0", zorder=3)
    for k, z in enumerate(roots):
        ax.annotate(f"$r_{k}$", (z.real * 1.25, z.imag * 1.25), ha="center", va="center")
        target = roots[perm[k]]
        if perm[k] != k:
            ax.annotate(
                "",
                xy=(target.real, target.imag),
                xytext=(z.real, z.imag),
                arrowprops=dict(arrowstyle="->", color="C3", connectionstyle="arc3,rad=0.25", shrinkA=6, shrinkB=6),
            )
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.set_title(str(perm), fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")
fig.suptitle(r"Galois group of $x^3-2$: the 6 permutations of its roots ($S_3$)")

# %%
# Which pairs commute?
# -----------------------------------------------------

els = galois.elements
commute = np.array([[galois.operate(a, b) == galois.operate(b, a) for b in els] for a in els])
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(commute, cmap="RdYlGn", vmin=0, vmax=1)
labels = [str(e) for e in els]
ax.set_xticks(range(len(els)), labels, rotation=45, fontsize=8)
ax.set_yticks(range(len(els)), labels, fontsize=8)
ax.set_title(r"$ab = ba$? (green: commute, red: do not)")
fig.tight_layout()
print(f"commuting ordered pairs: {commute.sum()} of {commute.size}")

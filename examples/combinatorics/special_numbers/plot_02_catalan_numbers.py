r"""
Catalan numbers: Euler's polygon triangulations and Catalan's brackets
===========================================================================

Draws all 14 triangulations of a convex hexagon, the problem Euler
posed to Goldbach in 1751, and checks that the same Catalan numbers
count balanced parenthesizations, Catalan's bracketing problem of 1838.
"""

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.combinatorics import catalan_number, combinations_count


def _triangulations(vertices):
    """All triangulations of the convex polygon ``vertices``, as lists of diagonals."""
    if len(vertices) < 3:
        return [[]]
    a, b = vertices[0], vertices[-1]
    result = []
    # The edge a-b lies in exactly one triangle (a, c, b); recurse on both sides.
    for i in range(1, len(vertices) - 1):
        c = vertices[i]
        chords = [(a, c)] if i > 1 else []
        chords += [(c, b)] if i < len(vertices) - 2 else []
        for left in _triangulations(vertices[: i + 1]):
            for right in _triangulations(vertices[i:]):
                result.append(chords + left + right)
    return result


def _is_balanced(s):
    depth = 0
    for ch in s:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


# %%
# Euler's problem: triangulations of a convex (n+2)-gon
# ---------------------------------------------------------------

for n in range(1, 8):
    count = len(_triangulations(list(range(n + 2))))
    print(f"{n + 2}-gon: {count:4d} triangulations, C_{n} = {catalan_number(n)}")
    assert count == catalan_number(n)

# %%
# Catalan's problem: balanced parenthesizations
# ---------------------------------------------------------------

for n in range(1, 7):
    balanced = sum(_is_balanced(s) for s in itertools.product("()", repeat=2 * n))
    closed_form = combinations_count(2 * n, n) // (n + 1)
    print(f"n={n}: {balanced:3d} balanced strings, C_{n} = {catalan_number(n)}, binom(2n,n)/(n+1) = {closed_form}")
print("n=3:", ["".join(s) for s in itertools.product("()", repeat=6) if _is_balanced(s)])

# %%
# The 14 triangulations of a hexagon
# ---------------------------------------------------------------

angles = np.pi / 2 + 2 * np.pi * np.arange(6) / 6
pts = np.column_stack([np.cos(angles), np.sin(angles)])
fig, axes = plt.subplots(2, 7, figsize=(12, 3.8))
for ax, diagonals in zip(axes.flat, _triangulations(list(range(6)))):
    ax.fill(pts[:, 0], pts[:, 1], color="C0", alpha=0.12)
    ax.plot(*np.vstack([pts, pts[:1]]).T, color="C0", lw=1.5)
    for i, j in diagonals:
        ax.plot(*pts[[i, j]].T, color="C3", lw=1.5)
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle(f"Euler's problem: a hexagon has $C_4 = {catalan_number(4)}$ triangulations")
fig.tight_layout()

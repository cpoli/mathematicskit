r"""
Heron's formula and needle-like triangles
===============================================

Checks Heron's formula against the shoelace formula, then shows why the
arrangement matters: for a thin "needle" triangle the textbook form
sqrt(s(s-a)(s-b)(s-c)) loses most of its digits, while Kahan's
rearrangement stays accurate.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.geometry import heron_area, polygon_area

# %%
# Agreement with the shoelace formula
# -----------------------------------------------------

tri = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]])
a, b, c = (np.linalg.norm(tri[i] - tri[(i + 1) % 3]) for i in range(3))
print(f"sides {a:.4f}, {b:.4f}, {c:.4f}: Heron {heron_area(a, b, c):.12f}, shoelace {polygon_area(tri):.12f}")

# %%
# Needle triangles: textbook form vs. Kahan's arrangement
# ------------------------------------------------------------


def textbook(a, b, c):
    s = (a + b + c) / 2
    return np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))


widths = np.logspace(-1, -12, 40)
exact = [w / 2 * np.sqrt(1 - w * w / 4) for w in widths]  # isosceles, legs of length 1, base w
naive_err = [abs(textbook(1.0, 1.0, w) - e) / e for w, e in zip(widths, exact)]
kahan_err = [max(abs(heron_area(1.0, 1.0, w) - e) / e, 1e-17) for w, e in zip(widths, exact)]

fig, ax = plt.subplots()
ax.loglog(widths, naive_err, "o-", ms=3, label="textbook Heron")
ax.loglog(widths, kahan_err, "s-", ms=3, label="Kahan's arrangement")
ax.invert_xaxis()
ax.set_xlabel("base of isosceles triangle with unit legs")
ax.set_ylabel("relative error in area")
ax.legend()

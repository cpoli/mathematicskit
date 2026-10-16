r"""
Lefschetz's fixed-point theorem and Hopf's trace formula (1926-1929)
====================================================================

Solomon Lefschetz assigned to every self-map :math:`f` of a complex the
number :math:`L(f) = \sum_k (-1)^k \operatorname{tr}(f_*: H_k \to H_k)`,
and proved that :math:`L(f) \ne 0` forces a fixed point. Hopf showed the
same alternating trace can be computed on chains, where it counts the
simplices :math:`f` maps to themselves. The identity has
:math:`L = \chi`. A rotation of the circle has :math:`L = 0` and indeed
moves every point. A reflection has :math:`L = 2`: its two fixed points.
The antipodal map of the sphere has :math:`L = 1 + (-1)^3 = 0`. The
converse fails: :math:`L = 0` does not rule fixed points out, as the
identity of the torus shows.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.topology import circle, lefschetz_number, sphere, torus

C = circle(8)
octahedron = sphere(2, "cross_polytope")
T = torus(4, 4)
maps = {
    "circle: identity": (C, list(range(8))),
    "circle: rotation by 1/8": (C, [(v + 1) % 8 for v in range(8)]),
    "circle: reflection": (C, [(-v) % 8 for v in range(8)]),
    "circle: collapse to a point": (C, [0] * 8),
    "sphere: identity": (octahedron, list(range(6))),
    "sphere: antipodal": (octahedron, [v ^ 1 for v in range(6)]),
    "sphere: reflection z -> -z": (octahedron, [v ^ 1 if v >= 4 else v for v in range(6)]),
    "torus: identity": (T, list(range(16))),
    "torus: translation": (T, [((v // 4 + 1) % 4) * 4 + v % 4 for v in range(16)]),
}
names, values = [], []
for name, (K, f) in maps.items():
    L = lefschetz_number(K, f)
    fixed = [v for v in range(len(f)) if f[v] == v]
    print(f"{name:>28}: L = {L:+d}   fixed vertices: {fixed}")
    names.append(name)
    values.append(L)

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.barh(names, values, color=["tab:red" if v else "0.6" for v in values])
ax.axvline(0, color="k", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("Lefschetz number L(f)")
ax.set_title("L(f) ≠ 0 forces a fixed point (red); L = 0 decides nothing")
ax.set_xticks(np.arange(-1, 4))

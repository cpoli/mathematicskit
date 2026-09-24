r"""
Euler and the Seven Bridges of Königsberg
===============================================

Reduces Königsberg to four land masses joined by seven bridges and
counts the bridges at each land mass. All four counts are odd, so by
Euler's argument no walk can cross every bridge exactly once, and an
exhaustive search agrees. Removing a single bridge leaves exactly two
odd land masses, and a walk over every remaining bridge appears.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import Graph, connected_components

# %%
# Land masses as vertices, bridges as edges
# -----------------------------------------------------
#
# The seven bridges form a multigraph: two pairs of bridges join the
# same land masses. :class:`~mathematicskit.graph_theory.core.base.Graph`
# stores simple graphs, so each bridge becomes its own vertex placed
# between the two banks it joins. A land mass's degree in this graph is
# then exactly the number of bridges that touch it.

land = ["A: north bank", "B: south bank", "C: Kneiphof island", "D: Lomse"]
bridges = [(0, 2), (0, 2), (1, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

g = Graph(len(land) + len(bridges))
for k, (u, v) in enumerate(bridges):
    g.add_edge(u, len(land) + k)
    g.add_edge(len(land) + k, v)

degrees = [len(g.neighbors(v)) for v in range(len(land))]
odd = [v for v, d in enumerate(degrees) if d % 2 == 1]
print(f"connected: {connected_components(g).n_components == 1}")
for name, d in zip(land, degrees):
    print(f"  {name:20s} {d} bridges ({'odd' if d % 2 else 'even'})")
print(f"odd land masses: {len(odd)} (an Euler walk allows at most 2)")


# %%
# Exhaustive search confirms Euler's argument
# -----------------------------------------------------


def euler_walks(bridge_list, n_land):
    """Every walk crossing each bridge exactly once, by depth-first search."""
    walks = []

    def extend(at, used, walk):
        if len(used) == len(bridge_list):
            walks.append(list(walk))
            return
        for k, (u, v) in enumerate(bridge_list):
            if k not in used and at in (u, v):
                walk.append(k)
                extend(v if at == u else u, used | {k}, walk)
                walk.pop()

    for start in range(n_land):
        extend(start, frozenset(), [])
    return walks


print(f"walks over all 7 bridges: {len(euler_walks(bridges, len(land)))}")

reduced = bridges[:-1]  # demolish the bridge between the island C and Lomse D
reduced_degrees = np.bincount(np.ravel(reduced), minlength=len(land))
walks = euler_walks(reduced, len(land))
print(f"after removing bridge C-D: degrees {reduced_degrees.tolist()}, {len(walks)} walks over all 6 bridges")
walk = walks[0]

# %%
# The map as a graph
# -----------------------------------------------------
#
# Red land masses touch an odd number of bridges. On the right, one
# bridge is removed and the numbers give the order of a walk that
# crosses every remaining bridge once, starting and ending at the two
# odd land masses.

pos = np.array([[0.0, 1.0], [0.0, -1.0], [-0.6, 0.0], [1.2, 0.0]])


def draw(ax, bridge_list, degs, order=None):
    seen = {}
    for k, (u, v) in enumerate(bridge_list):
        key = (min(u, v), max(u, v))
        count = sum(key == (min(a, b), max(a, b)) for a, b in bridge_list)
        i = seen.get(key, 0)
        seen[key] = i + 1
        bend = 0.0 if count == 1 else (0.25 if i == 0 else -0.25)
        p, q = pos[u], pos[v]
        normal = np.array([-(q - p)[1], (q - p)[0]])
        ctrl = (p + q) / 2 + bend * normal
        t = np.linspace(0, 1, 40)[:, None]
        curve = (1 - t) ** 2 * p + 2 * (1 - t) * t * ctrl + t**2 * q
        ax.plot(*curve.T, color="tab:brown", lw=3, zorder=1)
        if order is not None:
            mid = curve[20]
            ax.annotate(str(order.index(k) + 1), mid, ha="center", va="center", fontsize=9, bbox={"boxstyle": "circle", "fc": "white"}, zorder=3)
    colors = ["tab:red" if d % 2 else "tab:green" for d in degs]
    ax.scatter(*pos.T, s=900, c=colors, edgecolors="black", zorder=2)
    for v, (x, y) in enumerate(pos):
        ax.annotate(f"{'ABCD'[v]}\n{degs[v]}", (x, y), ha="center", va="center", color="white", fontweight="bold", zorder=3)
    ax.margins(0.15)
    ax.set_aspect("equal")
    ax.axis("off")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
draw(ax1, bridges, degrees)
ax1.set_title("Seven bridges: four odd land masses,\nno walk crosses each bridge once")
draw(ax2, reduced, reduced_degrees, order=walk)
ax2.set_title("Bridge C-D removed: two odd land masses,\nan Euler walk exists")
fig.tight_layout()

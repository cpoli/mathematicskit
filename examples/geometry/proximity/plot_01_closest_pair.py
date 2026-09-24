r"""
Shamos and Hoey: the closest pair of points
=================================================

Finds the closest pair in random point sets by divide and conquer,
checks the answer with a k-d tree, and compares the running time with
the brute-force comparison of every pair.
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from mathematicskit.geometry import closest_pair

# %%
# One point set
# -----------------------------------------------------

rng = np.random.default_rng(0)
points = rng.uniform(size=(400, 2))
result = closest_pair(points)
kd_distance = cKDTree(points).query(points, k=2)[0][:, 1].min()
print(f"closest pair {result.indices}, distance {result.distance:.6f} (k-d tree: {kd_distance:.6f})")

fig, ax = plt.subplots()
ax.plot(*points.T, ".", color="0.5")
ax.plot(*points[list(result.indices)].T, "o-", color="tab:red")
ax.set_aspect("equal")
ax.set_title("closest pair")

# %%
# Divide and conquer against brute force
# -----------------------------------------------------

sizes = [100, 200, 400, 800]
dc_times, brute_times = [], []
for n in sizes:
    pts = rng.uniform(size=(n, 2))
    start = time.perf_counter()
    closest_pair(pts)
    dc_times.append(time.perf_counter() - start)
    start = time.perf_counter()
    best = min(np.hypot(*(pts[i] - pts[j])) for i in range(n) for j in range(i + 1, n))
    brute_times.append(time.perf_counter() - start)

fig, ax = plt.subplots()
ax.loglog(sizes, dc_times, "o-", label=r"divide and conquer, $O(n \log n)$")
ax.loglog(sizes, brute_times, "s-", label=r"all pairs, $O(n^2)$")
ax.set_xlabel("number of points")
ax.set_ylabel("seconds")
ax.legend()

r"""
Kuhn's Hungarian method: the assignment problem
=====================================================

Assigns workers to jobs at minimum total cost. Brute force would check
n! assignments; the Hungarian method solves the problem in polynomial
time, and matches brute force on a small instance.
"""

# %%
import time
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

from mathematicskit.graph_theory import solve_assignment

# %%
# A small instance, checked by brute force
# -----------------------------------------------------

rng = np.random.default_rng(0)
cost = rng.integers(1, 30, size=(6, 6))
result = solve_assignment(cost)
brute = min(sum(cost[i, p[i]] for i in range(6)) for p in permutations(range(6)))
print(cost)
print(f"optimal assignment: {dict(zip(result.rows.tolist(), result.cols.tolist()))}, cost {result.total_cost} (brute force {brute})")

fig, ax = plt.subplots()
ax.imshow(cost, cmap="Blues")
ax.plot(result.cols, result.rows, "rx", ms=14, mew=3)
ax.set_xlabel("job")
ax.set_ylabel("worker")
ax.set_title("Chosen entries (red): one per row and column")

# %%
# Large instances are fast
# -----------------------------------------------------

for n in (100, 400, 1600):
    big = rng.random((n, n))
    start = time.perf_counter()
    total = solve_assignment(big).total_cost
    print(f"n = {n:4d}: optimal cost {total:.3f} in {time.perf_counter() - start:.3f} s ({n}! assignments)")

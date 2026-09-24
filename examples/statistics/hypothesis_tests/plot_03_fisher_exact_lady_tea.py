r"""
Fisher's exact test and the lady tasting tea
=================================================

Fisher's thought experiment: a lady claims she can tell whether milk or
tea was poured first. She is given 8 cups, 4 of each, and told so. If
she is only guessing, the number of milk-first cups she identifies
correctly follows a hypergeometric distribution, and getting all 4 right
happens with probability exactly 1/70.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mathematicskit.statistics import fisher_exact_test

# %%
# The null distribution of correct identifications
# -----------------------------------------------------

k = np.arange(5)
pmf = stats.hypergeom.pmf(k, 8, 4, 4)
for correct in range(5):
    table = [[correct, 4 - correct], [4 - correct, correct]]
    p = fisher_exact_test(table, alternative="greater").p_value
    print(f"{correct} of 4 correct: P(exactly) = {pmf[correct] * 70:.0f}/70, one-sided p = {p:.4f}")

fig, ax = plt.subplots()
ax.bar(k, pmf * 70)
ax.set_xlabel("milk-first cups correctly identified")
ax.set_ylabel("ways out of 70")
